import type {
  DecisionTrainingExample,
  DecisionTrainingSnapshotV1,
} from "./decision-training.js";

/**
 * Contract for a trainable corpus derived from `decision_training_examples`.
 *
 * The corpus is a build artifact, not a database table: `server` builds it from
 * captured rows, and evaluation/training consumers read the emitted JSONL. Both
 * sides assert these schema strings, so a format change fails closed on the
 * consumer instead of silently reading a shape it does not understand.
 */
export const DECISION_TRAINING_CORPUS_RECORD_SCHEMA = "paperclip.decision-training.corpus-record.v1" as const;
export const DECISION_TRAINING_CORPUS_MANIFEST_SCHEMA = "paperclip.decision-training.corpus-manifest.v1" as const;

/**
 * A credential-shaped match inside captured text. Carries the pattern name and
 * the JSON path only — never the matched value — so a rejection report cannot
 * itself become a second copy of the secret.
 */
export interface DecisionTrainingFinding {
  readonly pattern: string;
  readonly kind: "credential" | "injection";
  /** JSON path into the snapshot, for example `comments[2].body`. */
  readonly path: string;
}

export interface DecisionTrainingCorpusRecord {
  readonly schema: typeof DECISION_TRAINING_CORPUS_RECORD_SCHEMA;
  readonly id: string;
  readonly companyId: string;
  readonly issueId: string;
  readonly issueKey: string | null;
  readonly sourceKind: DecisionTrainingExample["sourceKind"];
  readonly sourceId: string;
  readonly cutoffAt: string;
  readonly retentionPolicy: DecisionTrainingExample["retentionPolicy"];
  readonly snapshot: DecisionTrainingSnapshotV1;
  readonly label: {
    readonly outcome: string | null;
    readonly notes: string;
    readonly codeResolution: DecisionTrainingSnapshotV1["code"]["resolution"];
    readonly hasCommitSha: boolean;
  };
  /**
   * Instruction-shaped text found in the record. Non-empty means the record
   * needs a human decision before it is trained on: the text is legitimate
   * negative signal in some runs and a captured injection attempt in others.
   */
  readonly injectionPatterns: readonly string[];
}

export interface DecisionTrainingCorpusSplit {
  readonly train: readonly DecisionTrainingCorpusRecord[];
  readonly holdout: readonly DecisionTrainingCorpusRecord[];
}

export interface DecisionTrainingCorpusManifest {
  readonly schema: typeof DECISION_TRAINING_CORPUS_MANIFEST_SCHEMA;
  readonly generatedAt: string;
  readonly source: {
    readonly kind: "decision_training_examples";
    readonly companyId: string | null;
    readonly rowCount: number;
    readonly retentionPolicy: DecisionTrainingExample["retentionPolicy"];
  };
  readonly split: {
    readonly salt: string;
    readonly holdoutRatio: number;
    readonly groups: number;
    readonly trainGroups: number;
    readonly holdoutGroups: number;
    readonly trainRecords: number;
    readonly holdoutRecords: number;
  };
  readonly distributions: {
    readonly sourceKind: Readonly<Record<string, number>>;
    readonly outcome: Readonly<Record<string, number>>;
    readonly codeResolution: Readonly<Record<string, number>>;
    readonly issueKey: Readonly<Record<string, number>>;
  };
  readonly bytes: {
    readonly train: number;
    readonly holdout: number;
  };
  readonly rejected: {
    readonly count: number;
    readonly reasons: Readonly<Record<string, number>>;
    readonly findings: readonly DecisionTrainingFinding[];
  };
  readonly warnings: readonly string[];
  /** Hashes pin the exact split so a later run can prove it is comparable. */
  readonly trainSha256: string;
  readonly holdoutSha256: string;
}
