/**
 * Corpus-readiness baseline for captured `decision_training_examples`.
 *
 * The question this answers is deliberately narrower than "is a model good?".
 * Before spending anything on fine-tuning, the question is whether the captured
 * data can support a training run at all: are the decisions labeled, is there a
 * rationale to imitate, is the resulting code traceable to a commit, is there
 * enough context to learn from, and is the text free of captured
 * instruction-shaped content.
 *
 * Everything here is deterministic and runs with no model and no credentials,
 * which is what makes it a baseline: the same corpus always produces the same
 * numbers, so a later change to the capture format or to the thresholds is
 * visible as a diff rather than hidden inside model variance.
 *
 * A model candidate plugs in as a second entry in the same matrix. The
 * scenarios, the scorer, and the thresholds stay fixed, so a candidate is only
 * ever compared against a corpus that a previous run also measured.
 */
import { runPaperclipEvalMatrix, type PaperclipEvalCandidate } from "@paperclipai/paperclip-eval-kernel";

/**
 * Mirrors `DECISION_TRAINING_CORPUS_RECORD_SCHEMA` from
 * `@paperclipai/shared`. The runner deliberately does not depend on that
 * package, so the schema string is asserted at the boundary instead of imported:
 * an unrecognized corpus fails closed here rather than being scored against
 * assumptions about a format it may not have.
 */
export const DECISION_TRAINING_CORPUS_EVAL_RECORD_SCHEMA = "paperclip.decision-training.corpus-record.v1" as const;
export const DECISION_TRAINING_CORPUS_BASELINE_REPORT_SCHEMA = "paperclip.runner.decision-training-corpus-baseline.v1" as const;

export type DecisionTrainingCodeResolution = "exact" | "nearest_run" | "workspace" | "none";

/** The only part of a corpus record this baseline reads. */
export interface DecisionTrainingCorpusEvalRecord {
  readonly schema: string;
  readonly id: string;
  readonly issueId: string;
  readonly sourceKind: string;
  readonly label: {
    readonly outcome: string | null;
    readonly notes: string;
    readonly codeResolution: DecisionTrainingCodeResolution;
    readonly hasCommitSha: boolean;
  };
  readonly snapshot: {
    readonly comments: readonly unknown[];
    readonly runs: readonly unknown[];
  };
  readonly injectionPatterns: readonly string[];
}

export const DECISION_TRAINING_BASELINE_CHECK_IDS = [
  "labeled_outcome",
  "operator_rationale",
  "verifiable_code",
  "commit_pinned",
  "context_sufficient",
  "no_injection_tags",
] as const;

export type DecisionTrainingBaselineCheckId = typeof DECISION_TRAINING_BASELINE_CHECK_IDS[number];

export interface DecisionTrainingCorpusBaselineThresholds {
  /** Minimum combined comment + run count for the record to carry signal. */
  readonly minContextItems: number;
  /** Minimum operator-note length, in characters, to count as a rationale. */
  readonly minRationaleChars: number;
}

export const DEFAULT_DECISION_TRAINING_CORPUS_BASELINE_THRESHOLDS: DecisionTrainingCorpusBaselineThresholds =
  Object.freeze({ minContextItems: 3, minRationaleChars: 20 });

export class DecisionTrainingCorpusBaselineError extends Error {
  readonly code = "paperclip_decision_training_corpus_baseline_invalid" as const;

  constructor(message: string) {
    super(message);
    this.name = "DecisionTrainingCorpusBaselineError";
  }
}

export interface DecisionTrainingCorpusEvalCheck {
  readonly id: DecisionTrainingBaselineCheckId;
  readonly passed: boolean;
  readonly detail: string;
}

export interface DecisionTrainingCorpusEvalScore {
  /** Share of checks passed, 0..1. Comparable across corpus revisions. */
  readonly readiness: number;
  /** Every check passed and the record has a usable learning target. */
  readonly trainable: boolean;
  /** Checks that block training, in catalog order. Empty when trainable. */
  readonly blockers: readonly DecisionTrainingBaselineCheckId[];
  readonly checks: readonly DecisionTrainingCorpusEvalCheck[];
}

export interface DecisionTrainingCorpusBaselineReport {
  readonly schema: typeof DECISION_TRAINING_CORPUS_BASELINE_REPORT_SCHEMA;
  readonly generatedAt: string;
  readonly thresholds: DecisionTrainingCorpusBaselineThresholds;
  readonly aggregate: {
    readonly records: number;
    readonly issues: number;
    readonly trainable: number;
    readonly trainableRate: number;
    readonly meanReadiness: number;
    readonly perCheck: Readonly<Record<DecisionTrainingBaselineCheckId, { passed: number; rate: number }>>;
    readonly blockers: Readonly<Record<string, number>>;
  };
  readonly distributions: {
    readonly sourceKind: Readonly<Record<string, number>>;
    readonly outcome: Readonly<Record<string, number>>;
    readonly codeResolution: Readonly<Record<string, number>>;
    readonly injectionPatterns: Readonly<Record<string, number>>;
  };
  readonly warnings: readonly string[];
  readonly records: ReadonlyArray<{
    readonly id: string;
    readonly issueId: string;
    readonly readiness: number;
    readonly trainable: boolean;
    readonly blockers: readonly DecisionTrainingBaselineCheckId[];
  }>;
}

/** The single deterministic candidate: the corpus exactly as captured. */
export const CAPTURED_CORPUS_BASELINE_CANDIDATE: PaperclipEvalCandidate<{ policy: "captured_as_is" }> = Object.freeze({
  id: "captured-corpus-baseline",
  config: Object.freeze({ policy: "captured_as_is" as const }),
});

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Fail-closed structural check. A corpus written by a different builder — or by
 * a hand edit that dropped the label block — is rejected outright rather than
 * scored, because a silently-degraded corpus produces plausible numbers that
 * mean nothing.
 */
export function assertDecisionTrainingCorpusEvalRecord(
  value: unknown,
  index: number,
): DecisionTrainingCorpusEvalRecord {
  if (!isRecord(value)) {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} is not an object`);
  }
  if (value.schema !== DECISION_TRAINING_CORPUS_EVAL_RECORD_SCHEMA) {
    throw new DecisionTrainingCorpusBaselineError(
      `corpus record ${index} has unsupported schema ${String(value.schema)}; expected ${DECISION_TRAINING_CORPUS_EVAL_RECORD_SCHEMA}`,
    );
  }
  for (const field of ["id", "issueId", "sourceKind"] as const) {
    if (typeof value[field] !== "string" || value[field].length === 0) {
      throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} is missing a string ${field}`);
    }
  }
  if (!isRecord(value.label)) {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} is missing its label`);
  }
  const label = value.label;
  if (typeof label.codeResolution !== "string") {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} label is missing codeResolution`);
  }
  if (typeof label.hasCommitSha !== "boolean") {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} label is missing hasCommitSha`);
  }
  if (label.outcome !== null && typeof label.outcome !== "string") {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} label outcome must be a string or null`);
  }
  if (typeof label.notes !== "string") {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} label notes must be a string`);
  }
  if (!isRecord(value.snapshot) || !Array.isArray(value.snapshot.comments) || !Array.isArray(value.snapshot.runs)) {
    throw new DecisionTrainingCorpusBaselineError(
      `corpus record ${index} snapshot must carry comments and runs arrays`,
    );
  }
  if (!Array.isArray(value.injectionPatterns)) {
    throw new DecisionTrainingCorpusBaselineError(`corpus record ${index} is missing injectionPatterns`);
  }
  return value as unknown as DecisionTrainingCorpusEvalRecord;
}

/** Parses JSONL text, skipping blank lines, and asserts every record. */
export function parseDecisionTrainingCorpusJsonl(text: string): DecisionTrainingCorpusEvalRecord[] {
  const records: DecisionTrainingCorpusEvalRecord[] = [];
  const lines = text.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]?.trim();
    if (!line) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch (error) {
      throw new DecisionTrainingCorpusBaselineError(
        `corpus line ${index + 1} is not valid JSON: ${(error as Error).message}`,
      );
    }
    records.push(assertDecisionTrainingCorpusEvalRecord(parsed, index));
  }
  return records;
}

function check(id: DecisionTrainingBaselineCheckId, passed: boolean, detail: string): DecisionTrainingCorpusEvalCheck {
  return { id, passed, detail };
}

export function scoreDecisionTrainingCorpusRecord(
  record: DecisionTrainingCorpusEvalRecord,
  thresholds: DecisionTrainingCorpusBaselineThresholds = DEFAULT_DECISION_TRAINING_CORPUS_BASELINE_THRESHOLDS,
): DecisionTrainingCorpusEvalScore {
  const { label } = record;
  const contextItems = record.snapshot.comments.length + record.snapshot.runs.length;
  const rationaleChars = label.notes.trim().length;

  const checks: readonly DecisionTrainingCorpusEvalCheck[] = [
    check(
      "labeled_outcome",
      typeof label.outcome === "string" && label.outcome.trim().length > 0,
      label.outcome === null || label.outcome.trim().length === 0
        ? "no decision outcome was recorded"
        : `outcome=${label.outcome}`,
    ),
    check(
      "operator_rationale",
      rationaleChars >= thresholds.minRationaleChars,
      `notes=${rationaleChars} chars (min ${thresholds.minRationaleChars})`,
    ),
    check(
      "verifiable_code",
      label.codeResolution === "exact" || label.codeResolution === "nearest_run",
      `codeResolution=${label.codeResolution}`,
    ),
    check("commit_pinned", label.hasCommitSha, label.hasCommitSha ? "commit sha present" : "no commit sha"),
    check(
      "context_sufficient",
      contextItems >= thresholds.minContextItems,
      `context=${contextItems} items (min ${thresholds.minContextItems})`,
    ),
    check(
      "no_injection_tags",
      record.injectionPatterns.length === 0,
      record.injectionPatterns.length === 0 ? "clean" : `flagged: ${record.injectionPatterns.join(", ")}`,
    ),
  ];

  const blockers = checks.filter((entry) => !entry.passed).map((entry) => entry.id);
  // A record is trainable only when it has a decision to imitate, a way to
  // check the answer, and enough context to generalize from. A high readiness
  // ratio on its own is not enough: a record with no label is not a training
  // example no matter how much text it carries.
  const trainable =
    blockers.length === 0 &&
    (label.outcome !== null || label.codeResolution === "exact" || label.codeResolution === "nearest_run");

  return {
    readiness: checks.length === 0 ? 0 : (checks.length - blockers.length) / checks.length,
    trainable,
    blockers,
    checks,
  };
}

function tally(values: readonly string[]): Readonly<Record<string, number>> {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return Object.fromEntries([...counts.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

function topKey(counts: Readonly<Record<string, number>>): string | undefined {
  return Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0];
}

function round(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

export async function runDecisionTrainingCorpusBaseline(input: {
  readonly records: readonly DecisionTrainingCorpusEvalRecord[];
  readonly thresholds?: DecisionTrainingCorpusBaselineThresholds;
  readonly now?: () => Date;
}): Promise<DecisionTrainingCorpusBaselineReport> {
  if (input.records.length === 0) {
    throw new DecisionTrainingCorpusBaselineError(
      "no corpus records were supplied; run the corpus builder before evaluating",
    );
  }
  const thresholds = input.thresholds ?? DEFAULT_DECISION_TRAINING_CORPUS_BASELINE_THRESHOLDS;
  const seenIds = new Set<string>();
  for (const [index, record] of input.records.entries()) {
    assertDecisionTrainingCorpusEvalRecord(record, index);
    if (seenIds.has(record.id)) {
      throw new DecisionTrainingCorpusBaselineError(`duplicate corpus record id: ${record.id}`);
    }
    seenIds.add(record.id);
  }

  const results = await runPaperclipEvalMatrix({
    scenarios: input.records.map((record) => ({ id: record.id, input: record })),
    candidates: [CAPTURED_CORPUS_BASELINE_CANDIDATE],
    execute: async ({ scenario }) => scenario.input,
    score: async ({ scenario, output }) => scoreDecisionTrainingCorpusRecord(output, thresholds),
  });

  const perCheck = Object.fromEntries(
    DECISION_TRAINING_BASELINE_CHECK_IDS.map((id) => {
      const entries = results.flatMap((result) => result.score.checks).filter((entry) => entry.id === id);
      const passed = entries.filter((entry) => entry.passed).length;
      return [id, { passed, rate: entries.length === 0 ? 0 : round(passed / entries.length) }];
    }),
  ) as Record<DecisionTrainingBaselineCheckId, { passed: number; rate: number }>;

  const blockers: string[] = [];
  const injectionPatterns: string[] = [];
  for (const result of results) {
    blockers.push(...result.score.blockers);
    for (const pattern of result.output.injectionPatterns) injectionPatterns.push(pattern);
  }

  const trainable = results.filter((result) => result.score.trainable).length;
  const meanReadiness = results.reduce((total, result) => total + result.score.readiness, 0) / results.length;
  const issueCount = new Set(input.records.map((record) => record.issueId)).size;

  const warnings: string[] = [];
  if (issueCount < 10) {
    warnings.push(
      `only ${issueCount} distinct issue(s) in this corpus; readiness numbers are stable but a train/holdout split needs more before any benchmark is meaningful`,
    );
  }
  if (trainable === 0) {
    warnings.push("no record passed every check; resolve the dominant blocker in `blockers` before capturing more data");
  }
  const dominantBlocker = topKey(tally(blockers));
  if (dominantBlocker) {
    warnings.push(`most common blocker: ${dominantBlocker}`);
  }

  return {
    schema: DECISION_TRAINING_CORPUS_BASELINE_REPORT_SCHEMA,
    generatedAt: (input.now?.() ?? new Date()).toISOString(),
    thresholds,
    aggregate: {
      records: results.length,
      issues: issueCount,
      trainable,
      trainableRate: round(trainable / results.length),
      meanReadiness: round(meanReadiness),
      perCheck,
      blockers: tally(blockers),
    },
    distributions: {
      sourceKind: tally(input.records.map((record) => record.sourceKind)),
      outcome: tally(input.records.map((record) => record.label.outcome ?? "(none)")),
      codeResolution: tally(input.records.map((record) => record.label.codeResolution)),
      injectionPatterns: tally(injectionPatterns),
    },
    warnings,
    records: results.map((result) => ({
      id: result.output.id,
      issueId: result.output.issueId,
      readiness: round(result.score.readiness),
      trainable: result.score.trainable,
      blockers: result.score.blockers,
    })),
  };
}
