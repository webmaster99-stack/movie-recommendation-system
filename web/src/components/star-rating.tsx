"use client";

import { useId, useState } from "react";
import { cn } from "@/lib/utils";

const STAR_PATH =
  "M12 2.6l2.84 5.98 6.56.87-4.8 4.56 1.2 6.51L12 17.38l-5.8 3.14 1.2-6.51-4.8-4.56 6.56-.87z";

function Star({ filled }: { filled: 0 | 0.5 | 1 }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className="size-full">
      <path d={STAR_PATH} className="fill-foreground/15" />
      <path
        d={STAR_PATH}
        className="fill-primary"
        style={{ clipPath: `inset(0 ${100 - filled * 100}% 0 0)` }}
      />
    </svg>
  );
}

const SIZES = { sm: "size-6", md: "size-8", lg: "size-10" };

type Props = {
  /** The current rating in half stars from 0.5 to 5; undefined when unrated. */
  value: number | undefined;
  onChange: (rating: number) => void;
  /** Names the movie for screen readers, e.g. "Rate Heat". */
  label: string;
  size?: keyof typeof SIZES;
  className?: string;
};

/**
 * Five stars, each split into two radio buttons, so half stars work with a mouse (left or
 * right half of a star), by touch and with the arrow keys.
 */
export function StarRating({ value, onChange, label, size = "md", className }: Props) {
  const name = useId();
  const [hovered, setHovered] = useState<number | undefined>(undefined);
  const shown = hovered ?? value ?? 0;

  return (
    <fieldset className={cn("flex", className)} onMouseLeave={() => setHovered(undefined)}>
      <legend className="sr-only">{label}</legend>
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className={cn(
            "relative rounded-sm has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring",
            SIZES[size],
          )}
        >
          <Star filled={shown >= star ? 1 : shown >= star - 0.5 ? 0.5 : 0} />
          {[star - 0.5, star].map((rating, half) => (
            <label
              key={rating}
              className={cn("absolute inset-y-0 w-1/2 cursor-pointer", half ? "right-0" : "left-0")}
              onMouseEnter={() => setHovered(rating)}
            >
              <input
                type="radio"
                name={name}
                value={rating}
                checked={value === rating}
                onChange={() => onChange(rating)}
                className="sr-only"
              />
              <span className="sr-only">{rating === 1 ? "1 star" : `${rating} stars`}</span>
            </label>
          ))}
        </span>
      ))}
    </fieldset>
  );
}

/** A rating shown, not asked for. */
export function Stars({ value, className }: { value: number; className?: string }) {
  return (
    <span className={cn("flex", className)} role="img" aria-label={`${value} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((star) => (
        <span key={star} className="size-4">
          <Star filled={value >= star ? 1 : value >= star - 0.5 ? 0.5 : 0} />
        </span>
      ))}
    </span>
  );
}
