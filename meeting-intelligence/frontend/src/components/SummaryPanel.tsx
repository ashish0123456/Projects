interface Props {
    summary: string | null;
}

export default function SummaryPanel({ summary }: Props) {
    if (!summary) {
        return (
            <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-lg flex items-center justify-center">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-bold text-white">Meeting Summary</h2>
                </div>
                <p className="text-white/60">No summary available yet.</p>
            </div>
        );
    }

    return (
        <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
            <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-lg flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                </div>
                <h2 className="text-xl font-bold text-white">Meeting Summary</h2>
            </div>
            <div className="prose prose-invert max-w-none">
                <p className="text-white/90 leading-relaxed whitespace-pre-wrap">{summary}</p>
            </div>
        </div>
    );
}