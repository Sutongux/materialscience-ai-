import React from "react";

export default function Header() {
  return (
    <header className="w-full text-center py-6 bg-gradient-to-r from-blue-500 to-teal-400 shadow-md rounded-lg">
      <h1 className="text-4xl font-bold text-white tracking-wide">
        Data Compounds
      </h1>
      <p className="text-white text-lg mt-2">
        Discover novel compounds for your goals
      </p>
    </header>
  );
}

