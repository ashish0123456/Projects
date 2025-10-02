import { useState, useEffect } from "react";
import client from "../api/client";

interface Progess {
    stage: string;
    data: any;
}

interface Props {
    meetingId: Number
}

export default function ProgressPanel({ meetingId }: Props) {
    const [progress, setProgress] = useState<Progess[]>([]);

    useEffect(() => {
        const fetchProgress = async () => {
            try {
                const res = await client.get(`/meetings/${meetingId}/progress`);
                setProgress(res.data);
            } catch (error) {
                console.error("Failed to fetch progress", error);
            }
        };

        const interval = setInterval(fetchProgress, 2000); // Poll every 2 seconds
        return () => clearInterval(interval);
    }, [meetingId]);

    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="font-semibold mb-2">Processing Progress</h2>
            <ul className="space-y-1 text-sm">
                {progress.map((item, index) => (
                    <li key={index} className="border-b pb-1">
                        {item.stage}
                    </li>
                ))}
            </ul>
        </div>
    );
}