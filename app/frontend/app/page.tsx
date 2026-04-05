"use client";

import { useRouter } from "next/navigation";

const stats = [
  { value: "24", label: "Signs Practiced", color: "text-violet-600" },
  { value: "68%", label: "Accuracy This Week", color: "text-violet-600" },
  { value: "3", label: "Sessions Completed", color: "text-violet-600" },
];

const weakSigns = ["Pain", "Appointment", "Help", "Medicine"];

export default function DashboardPage() {
  const router = useRouter();
  return (
    <div className="p-8 max-w-5xl mx-auto w-full">
      {/* Welcome */}
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-gray-900">Welcome Back, Maria</h1>
        <p className="text-gray-500 mt-1">
          You&apos;ve practiced 24 signs this week. Keep it up!
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-10">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-white rounded-2xl border border-gray-200 flex flex-col items-center justify-center py-8"
          >
            <span className={`text-4xl font-bold ${stat.color}`}>{stat.value}</span>
            <span className="text-gray-500 text-sm mt-1">{stat.label}</span>
          </div>
        ))}
      </div>

      {/* Today's Session */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6 flex items-center justify-between mb-8">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Today&apos;s Session — Healthcare Vocabulary
          </h2>
          <p className="text-gray-400 text-sm mt-0.5">10 signs &middot; ~8 minutes</p>
        </div>
        <button
          onClick={() => router.push("/practice")}
          className="bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold px-5 py-2.5 rounded-full transition-colors"
        >
          Start Session &rarr;
        </button>
      </div>

      {/* Weakest Signs */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">
          Weakest Signs (needs review)
        </h3>
        <div className="flex flex-wrap gap-3">
          {weakSigns.map((sign) => (
            <span
              key={sign}
              className="bg-pink-100 text-gray-800 text-sm font-medium px-6 py-2.5 rounded-full"
            >
              {sign}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
