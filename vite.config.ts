import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  plugins: [tailwindcss()],
  worker: { format: "es" },
  build: { target: "es2022" },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 60_000,
  },
});
