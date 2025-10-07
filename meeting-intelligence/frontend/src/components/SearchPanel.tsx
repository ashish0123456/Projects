import { useState } from "react";
import client from "../api/client";

interface Props {
    meetingId: number;
}

export default function SearchPanel({ meetingId }: Props) {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<any[]>([]);

    const handleSearch = async () => {
        try {
            const res = await client.post(`/meetings/${meetingId}/search/`, {
                query,
                top_k: 5,
            });
            setResults(res.data.results); // backend returns { query, results }
        } catch (err) {
            console.error("Search failed", err);
        }
    };

    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="font-semibold mb-2">Search Transcript</h2>
            <div className="flex mb-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="flex-1 border rounded-l px-2 py-1"
                    placeholder="Enter keyword or phrase"
                />
                <button
                    onClick={handleSearch}
                    className="bg-blue-600 text-white px-3 py-1 rounded-r hover:bg-blue-700"
                >
                    Search
                </button>
            </div>
            <ul className="list-disc list-inside space-y-1 text-sm">
                {results.map((r, idx) => (
                    <li key={idx}>
                        <b>{r.speaker_label}</b> ({r.start_time} - {r.end_time}): {r.text}
                    </li>
                ))}
            </ul>
        </div>
    );
}
