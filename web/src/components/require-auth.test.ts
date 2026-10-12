import { describe, expect, it } from "vitest";
import { destinationAfterLogin } from "./require-auth";

describe("destinationAfterLogin", () => {
  it("goes back to the page that asked for the login", () => {
    expect(destinationAfterLogin("?next=%2Fmovies%2F10")).toBe("/movies/10");
  });

  it("goes to the recommendations otherwise", () => {
    expect(destinationAfterLogin("")).toBe("/for-you");
  });

  it("never leaves the site, whatever the link says", () => {
    expect(destinationAfterLogin("?next=https%3A%2F%2Fevil.example")).toBe("/for-you");
    expect(destinationAfterLogin("?next=%2F%2Fevil.example")).toBe("/for-you");
  });
});
