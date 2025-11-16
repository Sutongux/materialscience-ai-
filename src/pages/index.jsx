import React, { useState } from "react";
import Header from "../components/Header.jsx";
import QueryBox from "../components/QueryBox.jsx";
import CompoundList from "../components/CompoundList.jsx";
import { getTopCompounds } from "../services/formulationEngine.js";

export default function HomePage() {
  const [compounds, setCompounds] = useState([]);

  const handleQuery = async (goal) => {
    const results = await getTopCompounds(goal);
    setCompounds(results);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
      <Header />
      <main className="w-full max-w-xl p-6 flex flex-col items-center mt-8">
        <QueryBox onSubmit={handleQuery} />
        <CompoundList compounds={compounds} />
      </main>
      <footer className="mt-auto py-6 text-center text-gray-500">
        © 2025 Data Compounds
      </footer>
    </div>
  );
}

