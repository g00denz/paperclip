/**
 * Build a reproducible, holdout-split training corpus from captured
 * `decision_training_examples`.
 *
 * This module is a pure transform: it takes the rows that
 * `decisionTrainingService.list` returns (or the JSONL emitted by
 * `GET /companies/:id/decision-training/export.jsonl`) and produces corpus
 * records plus a manifest. It never touches the database, so the same builder
 * runs in tests, in a one-shot script, and inside a future scheduled job.
 *
 * Two properties matter more than anything else here:
 *
 * 1. No secret ever reaches the corpus. Records whose free text matches a
 *    credential pattern are dropped from the corpus and reported as findings
 *    that name the JSON path and the pattern — never the matched value — so the
 *    operator can fix the source row without the report becoming a new leak.
 * 2. The holdout split is grouped by issue. Multiple examples routinely share
 *    one issue (an interaction, an approval, and the resulting execution
 *    decision are three rows describing the same thread). Splitting per row
 *    would put near-duplicates on both sides and silently inflate every later
 *    benchmark, so grouping is the default and cannot be turned off.
 */
import { createHash } from "node:crypto";

import {
  DECISION_TRAINING_CORPUS_MANIFEST_SCHEMA,
  DECISION_TRAINING_CORPUS_RECORD_SCHEMA,
  DECISION_TRAINING_RETENTION_POLICY,
  type DecisionTrainingCorpusManifest,
  type DecisionTrainingCorpusRecord,
  type DecisionTrainingCorpusSplit,
  type DecisionTrainingExample,
  type DecisionTrainingFinding,
  type DecisionTrainingSnapshotV1,
} from "@paperclipai/shared";

/** One redacted/free-text field is enough to hold a key, so scan every string. */
const MAX_SCANNED_STRING_LENGTH = 200_000;

interface CredentialPattern {
  readonly name: string;
  readonly pattern: RegExp;
}

/**
 * Deliberately narrow, high-signal credential shapes. A broad high-entropy
 * detector would fire on commit SHAs, base64 fixtures, and digests, so it is
 * not included; these are the patterns that are unambiguously credentials.
 */
const CREDENTIAL_PATTERNS: readonly CredentialPattern[] = Object.freeze([
  { name: "private_key_block", pattern: /-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----/ },
  { name: "anthropic_api_key", pattern: /\bsk-ant-[A-Za-z0-9_-]{20,}/ },
  { name: "openai_style_api_key", pattern: /\bsk-[A-Za-z0-9]{32,}/ },
  { name: "github_token", pattern: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}/ },
  { name: "github_fine_grained_token", pattern: /\bgithub_pat_[A-Za-z0-9_]{50,}/ },
  { name: "gitlab_token", pattern: /\bglpat-[A-Za-z0-9_-]{20,}/ },
  { name: "aws_access_key_id", pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/ },
  { name: "google_api_key", pattern: /\bAIza[0-9A-Za-z_-]{35}\b/ },
  { name: "slack_token", pattern: /\bxox[abposr]-[A-Za-z0-9-]{10,}/ },
  { name: "npm_token", pattern: /\bnpm_[A-Za-z0-9]{36}\b/ },
  { name: "stripe_secret_key", pattern: /\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}/ },
  { name: "sendgrid_api_key", pattern: /\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}/ },
  { name: "jwt", pattern: /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/ },
  { name: "authorization_header", pattern: /\b(?:Authorization|Proxy-Authorization)\s*[:=]\s*"?Bearer\s+[A-Za-z0-9._~+/-]{20,}/i },
]);

/**
 * Instruction-shaped text that an agent may have written into a comment or
 * result. These are not dropped: they are useful negative training signal. They
 * are tagged so a trainer can filter or deliberately train against them.
 */
const INJECTION_PATTERNS: readonly CredentialPattern[] = Object.freeze([
  { name: "ignore_previous_instructions", pattern: /ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions/i },
  { name: "disregard_instructions", pattern: /disregard\s+(?:all\s+)?(?:previous|prior|above)/i },
  { name: "credential_exfiltration", pattern: /(?:send|post|upload|exfiltrat\w*|email)\s+(?:the\s+)?(?:\.env|secrets?|api[_ -]?keys?|tokens?)\b/i },
  { name: "instruction_override_marker", pattern: /<\/?(?:system|assistant|im_start|im_end|important_instructions)>/i },
  { name: "remote_code_execution", pattern: /curl\s+[^\s|]*\|\s*(?:ba)?sh/ },
]);

export interface BuildDecisionTrainingCorpusInput {
  readonly examples: readonly DecisionTrainingExample[];
  /** Keep empty when the caller is a single-company export. */
  readonly companyId?: string | null;
  /** Changing the salt reshuffles the split without changing record content. */
  readonly holdoutSalt?: string;
  readonly holdoutRatio?: number;
  readonly now?: () => Date;
}

export interface BuildDecisionTrainingCorpusResult {
  readonly records: readonly DecisionTrainingCorpusRecord[];
  readonly split: DecisionTrainingCorpusSplit;
  readonly manifest: DecisionTrainingCorpusManifest;
}

export class DecisionTrainingCorpusError extends Error {
  readonly code = "paperclip_decision_training_corpus_invalid" as const;

  constructor(message: string) {
    super(message);
    this.name = "DecisionTrainingCorpusError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function snapshotIssueKey(snapshot: DecisionTrainingSnapshotV1): string | null {
  const key = snapshot.issue?.key;
  return typeof key === "string" && key.trim().length > 0 ? key : null;
}

/**
 * Walks arbitrary captured JSON and reports credential/injection matches. The
 * snapshot is `Record<string, unknown>` at every level, so a fixed field list
 * cannot be exhaustive; this generic walk is what makes the guarantee hold for
 * fields added to the capture later.
 */
function scanValue(
  value: unknown,
  path: string,
  findings: DecisionTrainingFinding[],
  seen: WeakSet<object>,
): void {
  if (typeof value === "string") {
    if (value.length === 0) return;
    const subject = value.length > MAX_SCANNED_STRING_LENGTH ? value.slice(0, MAX_SCANNED_STRING_LENGTH) : value;
    for (const { name, pattern } of CREDENTIAL_PATTERNS) {
      if (pattern.test(subject)) findings.push({ pattern: name, kind: "credential", path });
    }
    for (const { name, pattern } of INJECTION_PATTERNS) {
      if (pattern.test(subject)) findings.push({ pattern: name, kind: "injection", path });
    }
    return;
  }
  if (typeof value !== "object" || value === null) return;
  if (seen.has(value)) return;
  seen.add(value);
  if (Array.isArray(value)) {
    value.forEach((item, index) => scanValue(item, `${path}[${index}]`, findings, seen));
    return;
  }
  for (const [key, nested] of Object.entries(value)) {
    scanValue(nested, path ? `${path}.${key}` : key, findings, seen);
  }
}

function assertUsableSnapshot(example: DecisionTrainingExample): string | null {
  const snapshot: unknown = example.snapshot;
  if (!isRecord(snapshot)) return "snapshot_not_object";
  if (snapshot.version !== 1) return "unsupported_snapshot_version";
  if (typeof snapshot.capturedAt !== "string") return "missing_captured_at";
  if (!isRecord(snapshot.cutoff)) return "missing_cutoff";
  if (!isRecord(snapshot.issue)) return "missing_issue";
  if (!Array.isArray(snapshot.comments)) return "comments_not_array";
  if (!Array.isArray(snapshot.runs)) return "runs_not_array";
  if (!isRecord(snapshot.decision)) return "missing_decision";
  if (!isRecord(snapshot.code)) return "missing_code";
  if (typeof snapshot.code.resolution !== "string") return "missing_code_resolution";
  if (example.retentionPolicy !== DECISION_TRAINING_RETENTION_POLICY) return "unexpected_retention_policy";
  if (typeof example.id !== "string" || example.id.length === 0) return "missing_id";
  if (typeof example.issueId !== "string" || example.issueId.length === 0) return "missing_issue_id";
  return null;
}

function normalizeHoldoutRatio(value: number | undefined): number {
  const ratio = value ?? 0.1;
  if (!Number.isFinite(ratio) || ratio < 0 || ratio >= 1) {
    throw new DecisionTrainingCorpusError(`holdoutRatio must be in [0, 1); received ${String(value)}`);
  }
  return ratio;
}

/**
 * Deterministic 0..1 bucket from a stable hash. Using the salt as hash input
 * (rather than an ordering) keeps assignment independent of row order, so
 * re-exporting a later date does not reshuffle earlier records.
 */
function holdoutBucket(issueId: string, salt: string): number {
  const digest = createHash("sha256").update(`${salt}:${issueId}`).digest();
  return digest.readUInt32BE(0) / 0x1_0000_0000;
}

function tally(values: readonly (string | null)[]): Readonly<Record<string, number>> {
  const counts = new Map<string, number>();
  for (const value of values) {
    const key = value ?? "(none)";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Object.fromEntries([...counts.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

/**
 * Snapshots come from JSONB, so in practice they are acyclic. Callers can still
 * hand this builder a live object graph, and an unguarded `JSON.stringify` would
 * throw while measuring a corpus that the scanner already handled, so repeated
 * references are collapsed rather than allowed to abort the build.
 */
function safeStringify(value: unknown): string {
  const seen = new WeakSet<object>();
  return (
    JSON.stringify(value, (_key, nested: unknown) => {
      if (typeof nested !== "object" || nested === null) return nested;
      if (seen.has(nested)) return "[repeated]";
      seen.add(nested);
      return nested;
    }) ?? ""
  );
}

function sha256(records: readonly DecisionTrainingCorpusRecord[]): string {
  const hash = createHash("sha256");
  for (const record of records) {
    hash.update(record.id);
    hash.update("\n");
    hash.update(safeStringify(record));
    hash.update("\n");
  }
  return hash.digest("hex");
}

function byteLength(records: readonly DecisionTrainingCorpusRecord[]): number {
  let total = 0;
  for (const record of records) {
    total += Buffer.byteLength(safeStringify(record), "utf8");
  }
  return total;
}

export function buildDecisionTrainingCorpus(
  input: BuildDecisionTrainingCorpusInput,
): BuildDecisionTrainingCorpusResult {
  const holdoutRatio = normalizeHoldoutRatio(input.holdoutRatio);
  const salt = input.holdoutSalt ?? "paperclip-decision-training-v1";

  if (input.examples.length === 0) {
    throw new DecisionTrainingCorpusError(
      "no decision training examples were supplied; nothing to build. Capture examples via the decision training drawer before exporting.",
    );
  }

  const records: DecisionTrainingCorpusRecord[] = [];
  const rejectedReasons: Array<readonly [string, number]> = [];
  const rejectedCounts = new Map<string, number>();
  const findings: DecisionTrainingFinding[] = [];
  const seen = new WeakSet<object>();
  const companyIds = new Set<string>();
  let rowCount = 0;

  for (const example of input.examples) {
    rowCount += 1;
    const unusable = assertUsableSnapshot(example);
    if (unusable) {
      rejectedCounts.set(unusable, (rejectedCounts.get(unusable) ?? 0) + 1);
      continue;
    }
    if (input.companyId && example.companyId !== input.companyId) {
      rejectedCounts.set("company_mismatch", (rejectedCounts.get("company_mismatch") ?? 0) + 1);
      continue;
    }
    companyIds.add(example.companyId);

    const exampleFindings: DecisionTrainingFinding[] = [];
    scanValue(example.snapshot, "", exampleFindings, seen);
    scanValue(example.notes, "label.notes", exampleFindings, seen);

    if (exampleFindings.some((finding) => finding.kind === "credential")) {
      // Report the location and pattern, never the value.
      for (const finding of exampleFindings) {
        if (finding.kind === "credential") findings.push(finding);
      }
      rejectedCounts.set("credential_in_snapshot", (rejectedCounts.get("credential_in_snapshot") ?? 0) + 1);
      continue;
    }

    const snapshot = example.snapshot;
    records.push({
      schema: DECISION_TRAINING_CORPUS_RECORD_SCHEMA,
      id: example.id,
      companyId: example.companyId,
      issueId: example.issueId,
      issueKey: snapshotIssueKey(snapshot),
      sourceKind: example.sourceKind,
      sourceId: example.sourceId,
      cutoffAt: example.cutoffAt,
      retentionPolicy: example.retentionPolicy,
      snapshot,
      label: {
        outcome: example.decisionOutcome,
        notes: example.notes,
        codeResolution: snapshot.code.resolution,
        hasCommitSha: typeof snapshot.code.commitSha === "string" && snapshot.code.commitSha.length > 0,
      },
      injectionPatterns: [
        ...new Set(
          exampleFindings
            .filter((finding) => finding.kind === "injection")
            .map((finding) => finding.pattern),
        ),
      ].sort((a, b) => a.localeCompare(b)),
    });
  }

  if (records.length === 0) {
    throw new DecisionTrainingCorpusError(
      `all ${rowCount} rows were rejected (${[...rejectedCounts.entries()].map(([reason, count]) => `${reason}=${count}`).join(", ")}); no corpus was produced`,
    );
  }

  // Group by issue, then assign whole groups. Every example for an issue lands
  // on the same side of the split.
  const groups = new Map<string, DecisionTrainingCorpusRecord[]>();
  for (const record of records) {
    const bucket = groups.get(record.issueId);
    if (bucket) bucket.push(record);
    else groups.set(record.issueId, [record]);
  }

  const holdoutIssueIds = new Set<string>();
  for (const issueId of groups.keys()) {
    if (holdoutBucket(issueId, salt) < holdoutRatio) holdoutIssueIds.add(issueId);
  }

  // A holdout with no groups cannot be evaluated, and an empty train side makes
  // the corpus useless. Give each side at least one group once there is enough
  // data, otherwise report the gap instead of producing a broken split.
  const warnings: string[] = [];
  const issueIds = [...groups.keys()];
  if (holdoutIssueIds.size === 0 && issueIds.length > 1 && holdoutRatio > 0) {
    const promoted = issueIds
      .map((issueId) => ({ issueId, bucket: holdoutBucket(issueId, salt) }))
      .sort((a, b) => a.bucket - b.bucket)[0];
    if (promoted) holdoutIssueIds.add(promoted.issueId);
    warnings.push("holdoutRatio produced an empty holdout; the lowest-hash issue group was promoted so the split is evaluable");
  }
  if (holdoutIssueIds.size === groups.size && groups.size > 1) {
    const demoted = [...holdoutIssueIds][holdoutIssueIds.size - 1];
    if (demoted) holdoutIssueIds.delete(demoted);
    warnings.push("holdoutRatio captured every issue group; one group was returned to train so the corpus stays usable");
  }

  const train: DecisionTrainingCorpusRecord[] = [];
  const holdout: DecisionTrainingCorpusRecord[] = [];
  for (const [issueId, group] of groups) {
    (holdoutIssueIds.has(issueId) ? holdout : train).push(...group);
  }
  train.sort((a, b) => a.id.localeCompare(b.id));
  holdout.sort((a, b) => a.id.localeCompare(b.id));

  if (holdout.length === 0) {
    warnings.push(
      `only ${groups.size} issue group(s) are available, so the holdout is empty; capture more decisions before trusting any benchmark`,
    );
  }

  const rejectedCount = rowCount - records.length;
  for (const [reason, count] of rejectedCounts) rejectedReasons.push([reason, count]);

  const manifest: DecisionTrainingCorpusManifest = {
    schema: DECISION_TRAINING_CORPUS_MANIFEST_SCHEMA,
    generatedAt: (input.now?.() ?? new Date()).toISOString(),
    source: {
      kind: "decision_training_examples",
      companyId: companyIds.size === 1 ? [...companyIds][0] ?? null : null,
      rowCount,
      retentionPolicy: DECISION_TRAINING_RETENTION_POLICY,
    },
    split: {
      salt,
      holdoutRatio,
      groups: groups.size,
      trainGroups: groups.size - holdoutIssueIds.size,
      holdoutGroups: holdoutIssueIds.size,
      trainRecords: train.length,
      holdoutRecords: holdout.length,
    },
    distributions: {
      sourceKind: tally(records.map((record) => record.sourceKind)),
      outcome: tally(records.map((record) => record.label.outcome)),
      codeResolution: tally(records.map((record) => record.label.codeResolution)),
      issueKey: tally(records.map((record) => record.issueKey)),
    },
    bytes: {
      train: byteLength(train),
      holdout: byteLength(holdout),
    },
    rejected: {
      count: rejectedCount,
      reasons: Object.fromEntries(rejectedReasons.sort(([a], [b]) => a.localeCompare(b))),
      findings,
    },
    warnings,
    trainSha256: sha256(train),
    holdoutSha256: sha256(holdout),
  };

  return { records, split: { train, holdout }, manifest };
}
