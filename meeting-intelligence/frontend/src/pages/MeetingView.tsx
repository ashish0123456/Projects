import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import client from "../api/client";
import TranscriptPanel from "../components/TranscriptPanel";
import SummaryPanel from "../components/SummaryPanel";
import ActionItemsPanel from "../components/ActionItemsPanel";
import SearchPanel from "../components/SearchPanel";
import ProgressPanel from "../components/ProgressPanel";

interface MeetingData {
    transcript: string[];
    summary: string;
    actions: string[];
}

export default function MeetingView() {
    const { id } = useParams();
    const [data, setData] = useState<MeetingData | null>(null);
    const [loading, setLoading] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    const fetchMeetingDetails = async () => {
        try {
            setLoading(true);
            const res = await client.get(`/meetings/${id}/details/`);
            setData({
                transcript: res.data.transcript,
                summary: res.data.summary,
                actions: res.data.actions,
            });
        } catch (error) {
            console.error("Failed to load meeting details", error);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        if (isComplete) {
            fetchMeetingDetails();
        }
    }, [isComplete]);

    // While waiting for completion
    if (!isComplete || !data) {
        return (
            <div className="p-4 grid gap-4">
                <ProgressPanel meetingId={Number(id)} onComplete={() => setIsComplete(true)} />
                {loading && <p className="text-gray-600">Loading meeting details...</p>}
            </div>
        );
    }

    // After the completion of backgroud ingestion task
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <TranscriptPanel transcript={data.transcript} />
            <SummaryPanel summary={data.summary} />
            <ActionItemsPanel actions={data.actions} />
            <SearchPanel meetingId={Number(id)} />
        </div>
    );
}