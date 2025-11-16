import React, { useState } from "react";

export default function QueryBox({ onSubmit }) {
  const [goal, setGoal] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (goal.trim() !== "") {
      onSubmit(goal);
      setGoal("");
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col sm:flex-row items-center gap-4 w-full"
    >
      <input
        type="text"
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        placeholder="Enter your goal (e.g., lower freezing point of water)"
        className="flex-1 p-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 shadow-sm"
      />
      <button
        type="submit"
        className="px-6 py-3 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 transition-colors shadow-md"
      >
        Search
      </button>
    </form>
  );
}

