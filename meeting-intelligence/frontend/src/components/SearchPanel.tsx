import { useState } from "react";
import client from "../api/client";

interface Props {
    meetingId: number;
}

interface SearchResult {
    segment_id: number;
    start_time: number;
    end_time: number;
    speaker_label: string | null;
    text: string;
    score: number;
}

const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default function SearchPanel({ meetingId }: Props) {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<SearchResult[]>([]);
    const [searching, setSearching] = useState(false);

    const handleSearch = async () => {
        if (!query.trim()) return;
        setSearching(true);
        try {
            const res = await client.post(`/meetings/${meetingId}/search/`, {
                query,
                top_k: 5,
            });
            setResults(res.data.results || []);
        } catch (err) {
            console.error("Search failed", err);
            setResults([]);
        } finally {
            setSearching(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    return (
        <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
            <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-400 to-purple-400 rounded-lg flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                </div>
                <h2 className="text-xl font-bold text-white">Search Transcript</h2>
            </div>
            <div className="flex mb-4 space-x-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="flex-1 bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent"
                    placeholder="Enter keyword or phrase..."
                />
                <button
                    onClick={handleSearch}
                    disabled={searching || !query.trim()}
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-6 py-2 rounded-lg hover:from-indigo-600 hover:to-purple-600 transform hover:scale-105 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                    {searching ? (
                        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    ) : (
                        "Search"
                    )}
                </button>
            </div>
            {results.length > 0 && (
                <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                    <p className="text-white/60 text-sm mb-2">{results.length} result{results.length !== 1 ? 's' : ''} found</p>
                    {results.map((r, idx) => (
                        <div 
                            key={idx}
                            className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-all"
                        >
                            <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center space-x-2">
                                    {r.speaker_label && (
                                        <span className="px-3 py-1 bg-gradient-to-r from-indigo-500/30 to-purple-500/30 text-indigo-200 rounded-full text-xs font-semibold">
                                            {r.speaker_label}
                                        </span>
                                    )}
                                    <span className="text-white/50 text-xs">
                                        Score: {(r.score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <span className="text-white/50 text-xs font-mono">
                                    {formatTime(r.start_time)} - {formatTime(r.end_time)}
                                </span>
                            </div>
                            <p className="text-white/90 leading-relaxed">{r.text}</p>
                        </div>
                    ))}
                </div>
            )}
            {query && results.length === 0 && !searching && (
                <p className="text-white/60 text-center py-4">No results found. Try a different search term.</p>
            )}
        </div>
    );
}
