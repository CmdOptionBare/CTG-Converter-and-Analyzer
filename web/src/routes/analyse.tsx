import { createFileRoute } from "@tanstack/react-router";
import { AnalysisStation } from "@/components/ctg-analysis/AnalysisStation";

export const Route = createFileRoute("/analyse")({
  component: AnalysisStation,
});
