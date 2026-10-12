import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    rules: {
      // Posters are loaded straight from TMDB's image CDN with a plain <img>. next/image
      // would fetch them through our own server and keep resized copies there, which is
      // both a cost on the free plan and a second cache of TMDB's data to account for.
      "@next/next/no-img-element": "off",
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Generated from openapi.json by `npm run gen:api`.
    "src/lib/api/schema.d.ts",
  ]),
]);

export default eslintConfig;
