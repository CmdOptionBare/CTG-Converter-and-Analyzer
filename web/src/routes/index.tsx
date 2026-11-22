import { createFileRoute } from "@tanstack/react-router";
import { Station } from "@/components/ctg/station";

export const Route = createFileRoute("/")({ component: Station });
