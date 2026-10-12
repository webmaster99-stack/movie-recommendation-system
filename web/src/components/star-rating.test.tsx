import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StarRating } from "./star-rating";

describe("StarRating", () => {
  it("offers every half star from 0.5 to 5 under the movie's name", () => {
    render(<StarRating label="Rate Heat" value={undefined} onChange={() => {}} />);

    expect(screen.getByRole("group", { name: "Rate Heat" })).toBeInTheDocument();
    const names = screen.getAllByRole("radio").map((radio) => radio.getAttribute("value"));
    expect(names).toEqual(["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]);
  });

  it("shows the current rating as the selected option", () => {
    render(<StarRating label="Rate Heat" value={3.5} onChange={() => {}} />);

    expect(screen.getByRole("radio", { name: "3.5 stars" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "4 stars" })).not.toBeChecked();
  });

  it("reports a clicked rating, half stars included", async () => {
    const onChange = vi.fn();
    render(<StarRating label="Rate Heat" value={undefined} onChange={onChange} />);

    await userEvent.click(screen.getByRole("radio", { name: "4.5 stars" }));

    expect(onChange).toHaveBeenCalledExactlyOnceWith(4.5);
  });

  it("moves in half stars with the arrow keys", async () => {
    const onChange = vi.fn();
    render(<StarRating label="Rate Heat" value={3} onChange={onChange} />);

    screen.getByRole("radio", { name: "3 stars" }).focus();
    await userEvent.keyboard("{ArrowRight}");

    expect(onChange).toHaveBeenCalledExactlyOnceWith(3.5);
  });
});
