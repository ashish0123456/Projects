import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import client from "../api/client";
import TranscriptPanel from "../components/TranscriptPanel";
import SummaryPanel from "../components/SummaryPanel";
import ActionItemsPanel from "../components/ActionItemsPanel";
import SearchPanel from "../components/SearchPanel";

interface MeetingData {
    transcript: string[];
    summary: string;
    actions: string[];
}

export default function MeetingView() {
    const { id } = useParams();
    const [data, setData] = useState<MeetingData | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await client.get(`/meetings/${id}`);
                setData({
                    transcript: res.data.transcript,
                    summary: res.data.summary,
                    actions: res.data.actions
                });
            } catch (error) {
                console.error("Failed to load meeting", error);
            }
        };
        fetchData();
    }, [id]);

    if (!data) {
        return <p>Loading...</p>;
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <TranscriptPanel transcript={data.transcript} />
            <SummaryPanel summary={data.summary} />
            <ActionItemsPanel actions={data.actions} />
            <SearchPanel meetingId={Number(id)} />
        </div>
    );
}