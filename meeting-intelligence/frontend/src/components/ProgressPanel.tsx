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

    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="font-semibold mb-2">Processing Progress</h2>
            <ul className="space-y-1 text-sm">
                {progress && Object.keys(progress).length === 0 && (
                    <li>Checking progress...</li>
                )}

                {progress && (
                    <div className="space-y-2 text-sm">
                        <p>
                            <span className="font-medium text-gray-700">Status:</span>{" "}
                            <span className={`font-semibold ${progress.stage === "completed"
                                ? "text-green-600"
                                : "text-blue-600"
                                }`}>
                                {progress.stage}
                            </span>
                        </p>
                        {progress.detail?.msg && (
                            <p className="text-gray-600">
                                <span className="font-medium">Message:</span>{" "}
                                {progress.detail.msg}
                            </p>
                        )}
                    </div>
                )}

                {isComplete && (
                    <p className="text-green-600 font-semibold mt-3">
                        Processing completed successfully.
                    </p>
                )}
            </ul>
        </div>
    );
}