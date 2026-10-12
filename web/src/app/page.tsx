import Link from "next/link";
import { PosterWall } from "@/components/poster-wall";
import { StartButton } from "@/components/start-button";
import { Button } from "@/components/ui/button";

const STEPS = [
  {
    title: "Rate ten movies you've seen",
    text: "Half a star to five. Well-known titles from every genre come first, so there is something you recognise whatever you watch.",
  },
  {
    title: "Get twenty to watch next",
    text: "The model compares what you liked with the ratings of 72,000 other people and ranks the whole catalog for you. It takes about five milliseconds.",
  },
  {
    title: "Keep rating, and the list keeps up",
    text: "Nothing is retrained and nothing waits for tonight's batch job. The rating you gave a second ago already counts.",
  },
];

export default function LandingPage() {
  return (
    <>
      <section className="container-page pt-14 pb-12 sm:pt-20">
        <h1 className="max-w-4xl text-6xl leading-[0.9] font-extrabold uppercase sm:text-8xl">
          Rate ten movies. Get twenty worth watching.
        </h1>
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
          A recommender trained on 8.5 million ratings from MovieLens. Tell it what you liked and it
          picks from nearly 20,000 movies.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <StartButton />
          <Button variant="ghost" className="h-12 px-4 text-base" asChild>
            <Link href="/about">How it works</Link>
          </Button>
        </div>
      </section>

      <PosterWall />

      <section className="container-page mt-20">
        <ol className="grid gap-x-10 gap-y-10 md:grid-cols-3">
          {STEPS.map((step, index) => (
            <li key={step.title}>
              <span className="font-heading text-6xl leading-none font-extrabold text-primary">
                {index + 1}
              </span>
              <h2 className="mt-3 text-3xl leading-tight font-bold">{step.title}</h2>
              <p className="mt-3 leading-relaxed text-muted-foreground">{step.text}</p>
            </li>
          ))}
        </ol>
      </section>
    </>
  );
}
