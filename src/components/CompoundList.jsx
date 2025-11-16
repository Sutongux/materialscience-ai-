import React from "react";

export default function CompoundList({ compounds }) {
  if (!compounds || compounds.length === 0) {
    return (
      <p className="text-center text-gray-500 mt-4">
        No compounds to display yet.
      </p>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-2 mt-4">
      {compounds.map((compound, index) => (
        <div
          key={index}
          className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow"
        >
          <h2 className="text-xl font-semibold text-blue-600">
            {compound.name}
          </h2>
          <p className="text-gray-700 mt-2">{compound.summary}</p>
        </div>
      ))}
    </div>
  );
}

