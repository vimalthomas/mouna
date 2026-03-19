import { InferenceResult, Sign } from "@/lib/types";

interface Props {
  result: InferenceResult | null;
  signs: Sign[];
  idle?: boolean;
}

export default function Results({ result, signs, idle = false }: Props) {
  if (idle || !result) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center gap-3 py-12">
        <span className="text-5xl">🤟</span>
        <p className="text-zinc-500 text-sm">Press Record and perform a sign</p>
      </div>
    );
  }

  const matchedSign = signs.find((s) => s.word === result.prediction);
  const pct = Math.round(result.confidence * 100);

  return (
    <div className="flex flex-col gap-5">
      <div>
        <p className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Prediction</p>
        {result.uncertain ? (
          <p className="text-2xl font-bold text-zinc-400">Uncertain</p>
        ) : (
          <p className="text-3xl font-bold text-white capitalize">{result.prediction}</p>
        )}
        {matchedSign && (
          <p className="text-sm text-zinc-400 mt-1">{matchedSign.description}</p>
        )}
      </div>

      <div>
        <div className="flex justify-between text-xs text-zinc-400 mb-1">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              result.uncertain ? "bg-zinc-600" : pct > 70 ? "bg-emerald-500" : pct > 40 ? "bg-amber-500" : "bg-red-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div>
        <p className="text-xs text-zinc-500 uppercase tracking-widest mb-2">Top 3</p>
        <div className="flex flex-col gap-1.5">
          {result.top3.map((p, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs text-zinc-600 w-4">{i + 1}.</span>
              <span className="text-sm text-zinc-300 capitalize w-24 truncate">{p.label}</span>
              <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-600 rounded-full"
                  style={{ width: `${Math.round(p.confidence * 100)}%` }}
                />
              </div>
              <span className="text-xs text-zinc-500 w-8 text-right">
                {Math.round(p.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
