import { expect, it } from "vitest";

import {
  DEFAULT_DECISION_TRAINING_CORPUS_BASELINE_THRESHOLDS,
  DECISION_TRAINING_CORPUS_EVAL_RECORD_SCHEMA,
  assertDecisionTrainingCorpusEvalRecord,
  parseDecisionTrainingCorpusJsonl,
  runDecisionTrainingCorpusBaseline,
  scoreDecisionTrainingCorpusRecord,
  type DecisionTrainingCorpusEvalRecord,
} from "./decision-training-corpus-baseline.js";

const FIXED_NOW = () => new Date("2026-04-01T00:00:00.000Z");

function record(overrides: Partial<DecisionTrainingCorpusEvalRecord> = {}): DecisionTrainingCorpusEvalRecord {
  return {
    schema: DECISION_TRAINING_CORPUS_EVAL_RECORD_SCHEMA,
    id: "example-1",
    issueId: "issue-1",
    sourceKind: "interaction",
    label: {
      outcome: "approved",
      notes: "Retry with exponential backoff and cap the attempt budget.",
      codeResolution: "exact",
      hasCommitSha: true,
    },
    snapshot: { comments: [{ body: "a" }, { body: "b" }], runs: [{ id: "run-1" }] },
    injectionPatterns: [],
    ...overrides,
  };
}

it("scores a fully specified record as trainable", () => {
  const score = scoreDecisionTrainingCorpusRecord(record());
  expect(score.trainable).toBe(true);
  expect(score.readiness).toBe(1);
  expect(score.blockers).toEqual([]);
  expect(score.checks.map((entry) => entry.id)).toEqual([
    "labeled_outcome",
    "operator_rationale",
    "verifiable_code",
    "commit_pinned",
    "context_sufficient",
    "no_injection_tags",
  ]);
});

it("treats an unlabeled record with plenty of text as not trainable", () => {
  const score = scoreDecisionTrainingCorpusRecord(
    record({
      label: {
        outcome: null,
        notes: "A long and thoughtful operator rationale that is not attached to any decision outcome.",
        codeResolution: "none",
        hasCommitSha: false,
      },
    }),
  );
  expect(score.trainable).toBe(false);
  expect(score.blockers).toContain("labeled_outcome");
  expect(score.blockers).toContain("verifiable_code");
  // Half the checks still pass, which is exactly why readiness alone is not a
  // gate: without a decision and a verifiable target there is nothing to learn.
  expect(score.readiness).toBe(0.5);
});

it("blocks a record carrying instruction-shaped text", () => {
  const score = scoreDecisionTrainingCorpusRecord(record({ injectionPatterns: ["ignore_previous_instructions"] }));
  expect(score.trainable).toBe(false);
  expect(score.blockers).toEqual(["no_injection_tags"]);
  expect(score.checks.at(-1)?.detail).toContain("ignore_previous_instructions");
});

it("respects custom thresholds", () => {
  const sparse = record({ snapshot: { comments: [], runs: [{ id: "run-1" }] } });
  expect(scoreDecisionTrainingCorpusRecord(sparse).blockers).toContain("context_sufficient");
  expect(
    scoreDecisionTrainingCorpusRecord(sparse, { minContextItems: 1, minRationaleChars: 20 }).blockers,
  ).not.toContain("context_sufficient");
  expect(
    scoreDecisionTrainingCorpusRecord(record(), { minContextItems: 1, minRationaleChars: 5_000 }).blockers,
  ).toContain("operator_rationale");
  expect(DEFAULT_DECISION_TRAINING_CORPUS_BASELINE_THRESHOLDS.minContextItems).toBe(3);
});

it("fails closed on an unknown or malformed corpus record", () => {
  expect(() => assertDecisionTrainingCorpusEvalRecord(record({ schema: "other.v9" }), 0)).toThrow(
    /unsupported schema other.v9/,
  );
  expect(() => assertDecisionTrainingCorpusEvalRecord(null, 0)).toThrow(/not an object/);
  expect(() => assertDecisionTrainingCorpusEvalRecord({ ...record(), label: undefined }, 0)).toThrow(/missing its label/);
  expect(() => assertDecisionTrainingCorpusEvalRecord({ ...record(), snapshot: { comments: [] } }, 0)).toThrow(
    /comments and runs arrays/,
  );
  expect(() => assertDecisionTrainingCorpusEvalRecord({ ...record(), injectionPatterns: "none" }, 0)).toThrow(
    /missing injectionPatterns/,
  );
});

it("parses JSONL, skipping blank lines and reporting the offending line", () => {
  const parsed = parseDecisionTrainingCorpusJsonl(
    `${JSON.stringify(record({ id: "a" }))}\n\n  \n${JSON.stringify(record({ id: "b" }))}\n`,
  );
  expect(parsed.map((entry) => entry.id)).toEqual(["a", "b"]);
  expect(() => parseDecisionTrainingCorpusJsonl("{not json}\n")).toThrow(/line 1 is not valid JSON/);
  expect(() => parseDecisionTrainingCorpusJsonl('{"schema":"x"}\n')).toThrow(/unsupported schema x/);
});

it("aggregates readiness, blockers, and distributions across a corpus", async () => {
  const report = await runDecisionTrainingCorpusBaseline({
    records: [
      record({ id: "a", issueId: "issue-1" }),
      record({ id: "b", issueId: "issue-2", injectionPatterns: ["remote_code_execution"] }),
      record({
        id: "c",
        issueId: "issue-2",
        sourceKind: "approval",
        label: { outcome: null, notes: "", codeResolution: "none", hasCommitSha: false },
      }),
    ],
    now: FIXED_NOW,
  });

  expect(report.schema).toBe("paperclip.runner.decision-training-corpus-baseline.v1");
  expect(report.generatedAt).toBe("2026-04-01T00:00:00.000Z");
  expect(report.aggregate.records).toBe(3);
  expect(report.aggregate.issues).toBe(2);
  expect(report.aggregate.trainable).toBe(1);
  expect(report.aggregate.trainableRate).toBeCloseTo(1 / 3, 4);
  expect(report.aggregate.perCheck.no_injection_tags).toEqual({ passed: 2, rate: 0.6667 });
  expect(report.aggregate.perCheck.labeled_outcome).toEqual({ passed: 2, rate: 0.6667 });
  expect(report.aggregate.blockers.operator_rationale).toBe(1);
  expect(report.aggregate.blockers.no_injection_tags).toBe(1);
  expect(report.distributions.sourceKind).toEqual({ approval: 1, interaction: 2 });
  expect(report.distributions.outcome).toEqual({ "(none)": 1, approved: 2 });
  expect(report.distributions.codeResolution).toEqual({ exact: 2, none: 1 });
  expect(report.distributions.injectionPatterns).toEqual({ remote_code_execution: 1 });
  expect(report.records.map((entry) => entry.id)).toEqual(["a", "b", "c"]);
  expect(report.records[0]).toEqual({ id: "a", issueId: "issue-1", readiness: 1, trainable: true, blockers: [] });
});

it("warns when a corpus is too small or has nothing trainable in it", async () => {
  const small = await runDecisionTrainingCorpusBaseline({ records: [record()], now: FIXED_NOW });
  expect(small.warnings.join(" ")).toContain("only 1 distinct issue");

  const unusable = await runDecisionTrainingCorpusBaseline({
    records: Array.from({ length: 12 }, (_, index) =>
      record({
        id: `example-${index}`,
        issueId: `issue-${index}`,
        label: { outcome: null, notes: "", codeResolution: "none", hasCommitSha: false },
      }),
    ),
    now: FIXED_NOW,
  });
  expect(unusable.aggregate.trainable).toBe(0);
  expect(unusable.warnings.join(" ")).toContain("no record passed every check");
  expect(unusable.warnings.join(" ")).toContain("most common blocker");
  // Only context_sufficient and no_injection_tags pass on a fully unlabeled row.
  expect(unusable.aggregate.meanReadiness).toBeCloseTo(2 / 6, 4);
  expect(unusable.aggregate.blockers.labeled_outcome).toBe(12);
});

it("refuses an empty corpus and duplicate record ids", async () => {
  await expect(runDecisionTrainingCorpusBaseline({ records: [] })).rejects.toThrow(
    /run the corpus builder before evaluating/,
  );
  await expect(
    runDecisionTrainingCorpusBaseline({ records: [record({ id: "dup" }), record({ id: "dup", issueId: "issue-2" })] }),
  ).rejects.toThrow(/duplicate corpus record id: dup/);
});
