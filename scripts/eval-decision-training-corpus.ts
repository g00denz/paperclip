#!/usr/bin/env node
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { buildDecisionTrainingCorpus } from "../server/src/services/decision-training-corpus.js";
import {
  parseDecisionTrainingCorpusJsonl,
  runDecisionTrainingCorpusBaseline,
} from "../packages/paperclip-runner/src/eval/decision-training-corpus-baseline.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

function readJson(line: string, lineNumber: number): unknown {
  try {
    return JSON.parse(line);
  } catch (error) {
    throw new Error(`corpus line ${lineNumber} is not valid JSON: ${(error as Error).message}`);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes("--help") || args.includes("-h")) {
    console.log("Usage: tsx scripts/eval-decision-training-corpus.ts [--company <id>] [--holdout-ratio 0.1] [--input data.jsonl] [--output-dir evals/decision-training]");
    process.exit(0);
  }
  const getArg = (key: string): string | undefined => {
    const index = args.indexOf(key);
    return index >= 0 && args[index + 1] ? args[index + 1] : undefined;
  };
  const companyId = getArg("--company") ?? null;
  const holdoutRatio = getArg("--holdout-ratio") ? Number(getArg("--holdout-ratio")) : 0.1;
  const inputPath = getArg("--input") ?? resolve(ROOT, "evals/decision-training/corpus-raw.jsonl");
  const outputDir = getArg("--output-dir") ?? resolve(ROOT, "evals/decision-training");

  if (!existsSync(inputPath)) {
    throw new Error(`no input corpus found at ${inputPath}. Create evals/decision-training/corpus-raw.jsonl from a company export.`);
  }
  const lines = readFileSync(inputPath, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const examples = lines.map((line, index) => readJson(line, index + 1));
  const built = buildDecisionTrainingCorpus({
    examples: examples as Parameters<typeof buildDecisionTrainingCorpus>[0]["examples"],
    companyId,
    holdoutRatio,
  });

  const recordLines = built.records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  const trainLines = built.split.train.map((record) => JSON.stringify(record)).join("\n") + "\n";
  const holdoutLines = built.split.holdout.map((record) => JSON.stringify(record)).join("\n") + "\n";

  writeFileSync(resolve(outputDir, "corpus-all.jsonl"), recordLines);
  writeFileSync(resolve(outputDir, "corpus-train.jsonl"), trainLines);
  writeFileSync(resolve(outputDir, "corpus-holdout.jsonl"), holdoutLines);
  writeFileSync(resolve(outputDir, "corpus-manifest.json"), JSON.stringify(built.manifest, null, 2) + "\n");

  const baseline = await runDecisionTrainingCorpusBaseline({
    records: parseDecisionTrainingCorpusJsonl(recordLines),
  });
  writeFileSync(resolve(outputDir, "baseline-corpus.json"), JSON.stringify(baseline, null, 2) + "\n");

  console.log(
    JSON.stringify({
      records: built.records.length,
      train: built.split.train.length,
      holdout: built.split.holdout.length,
      trainable: baseline.aggregate.trainable,
      meanReadiness: baseline.aggregate.meanReadiness,
    }),
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
