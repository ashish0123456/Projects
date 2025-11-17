interface TranscriptSegment {
    id: number;
    meeting_id: number;
    start_time: number;
    end_time: number;
    speaker: string | null;
    text: string;
}

interface Props {
    transcript: TranscriptSegment[];
}

const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default function TranscriptPanel({ transcript }: Props) {
    if (transcript.length === 0) {
        return (
            <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-green-400 to-emerald-400 rounded-lg flex items-center justify-center">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-bold text-white">Transcript</h2>
                </div>
                <p className="text-white/60">No transcript available yet.</p>
            </div>
        );
    }

    return (
        <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-green-400 to-emerald-400 rounded-lg flex items-center justify-center">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-bold text-white">Transcript</h2>
                </div>
                <span className="text-white/60 text-sm">{transcript.length} segments</span>
            </div>
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                {transcript.map((segment) => (
                    <div
                        key={segment.id}
                        className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-all"
                    >
                        <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                                {segment.speaker && (
                                    <span className="px-3 py-1 bg-gradient-to-r from-purple-500/30 to-pink-500/30 text-purple-200 rounded-full text-xs font-semibold">
                                        {segment.speaker}
                                    </span>
                                )}
                            </div>
                            <span className="text-white/50 text-xs font-mono">
                                {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
                            </span>
                        </div>
                        <p className="text-white/90 leading-relaxed">{segment.text}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}