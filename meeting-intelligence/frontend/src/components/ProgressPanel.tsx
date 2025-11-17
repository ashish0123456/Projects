import { useState, useEffect } from "react";
import client from "../api/client";

interface Progress {
    stage: string;
    meeting_id: number;
    detail: { msg?: string };
}

interface Props {
    meetingId: Number;
    onComplete: () => void; // callback to notify parent when task is completed
}

const getStageInfo = (stage: string) => {
    const stages: Record<string, { label: string; icon: string; color: string }> = {
        'uploaded': { label: 'Uploaded', icon: '📤', color: 'from-blue-400 to-cyan-400' },
        'transcribing': { label: 'Transcribing', icon: '🎤', color: 'from-purple-400 to-pink-400' },
        'diarizing': { label: 'Identifying Speakers', icon: '👥', color: 'from-indigo-400 to-purple-400' },
        'summarizing': { label: 'Generating Summary', icon: '📝', color: 'from-green-400 to-emerald-400' },
        'extracting': { label: 'Extracting Action Items', icon: '✅', color: 'from-orange-400 to-red-400' },
        'completed': { label: 'Completed', icon: '✨', color: 'from-green-400 to-teal-400' },
    };
    return stages[stage] || { label: stage, icon: '⏳', color: 'from-gray-400 to-gray-500' };
};

export default function ProgressPanel({ meetingId, onComplete }: Props) {
    const [progress, setProgress] = useState<Progress | null>(null);
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (!meetingId) return;

        const fetchProgress = async () => {
            try {
                const res = await client.get(`/meetings/${meetingId}/progress/`);
                const data = res.data;
                setProgress(data);

                if (data?.stage === "completed") {
                    setIsComplete(true);
                    onComplete(); // notify parent about the process completion
                }
            } catch (error) {
                console.error("Failed to fetch progress", error);
            }
        };

        fetchProgress(); // fetch immediately on mount
        const interval = setInterval(fetchProgress, 2000); // Poll every 2 seconds

        return () => clearInterval(interval);
    }, [meetingId, onComplete]);

    const stageInfo = progress ? getStageInfo(progress.stage) : null;

    return (
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-8 shadow-xl">
            <div className="flex items-center space-x-3 mb-6">
                <div className="w-12 h-12 bg-gradient-to-br from-purple-400 to-pink-400 rounded-lg flex items-center justify-center">
                    <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                </div>
                <h2 className="text-2xl font-bold text-white">Processing Progress</h2>
            </div>

            {!progress || Object.keys(progress).length === 0 ? (
                <div className="flex items-center justify-center py-8">
                    <div className="flex items-center space-x-3">
                        <svg className="animate-spin h-6 w-6 text-purple-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <p className="text-white/70">Checking progress...</p>
                    </div>
                </div>
            ) : (
                <div className="space-y-6">
                    <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                        <div className="flex items-center space-x-4 mb-4">
                            <div className={`w-16 h-16 bg-gradient-to-br ${stageInfo?.color} rounded-xl flex items-center justify-center text-3xl shadow-lg`}>
                                {stageInfo?.icon}
                            </div>
                            <div className="flex-1">
                                <p className="text-white/60 text-sm mb-1">Current Stage</p>
                                <p className={`text-2xl font-bold ${
                                    progress.stage === "completed" 
                                        ? "text-green-400" 
                                        : "text-white"
                                }`}>
                                    {stageInfo?.label}
                                </p>
                            </div>
                        </div>
                        {progress.detail?.msg && (
                            <p className="text-white/80 leading-relaxed">
                                {progress.detail.msg}
                            </p>
                        )}
                    </div>

                    {isComplete && (
                        <div className="bg-gradient-to-r from-green-500/20 to-teal-500/20 border border-green-400/30 rounded-xl p-6">
                            <div className="flex items-center space-x-3">
                                <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <p className="text-green-300 font-semibold text-lg">
                                    Processing completed successfully!
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}