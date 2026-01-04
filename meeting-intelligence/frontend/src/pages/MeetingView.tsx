import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import client from "../api/client";
import TranscriptPanel from "../components/TranscriptPanel";
import SummaryPanel from "../components/SummaryPanel";
import ActionItemsPanel from "../components/ActionItemsPanel";
import SearchPanel from "../components/SearchPanel";
import ProgressPanel from "../components/ProgressPanel";

interface TranscriptSegment {
    id: number;
    meeting_id: number;
    start_time: number;
    end_time: number;
    speaker: string | null;
    text: string;
}

interface ActionItem {
    id: number;
    meeting_id: number;
    segment_id: number | null;
    text: string;
    assigned_to: string | null;
    resolved: boolean;
}

export default function MeetingView() {
    const { id } = useParams();
    const [meetingInfo, setMeetingInfo] = useState<any>(null);
    const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
    const [summary, setSummary] = useState<string | null>(null);
    const [actions, setActions] = useState<ActionItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    const fetchMeetingInfo = async () => {
        try {
            const res = await client.get(`/meetings/${id}/`);
            setMeetingInfo(res.data);
        } catch (error) {
            console.error("Failed to load meeting info", error);
        }
    };

    const fetchMeetingDetails = async () => {
        try {
            setLoading(true);
            // Fetch transcript segments with full data
            const transcriptRes = await client.get(`/meetings/${id}/transcripts/`);
            setTranscript(transcriptRes.data);

            // Fetch summary
            try {
                const summaryRes = await client.get(`/meetings/${id}/summary/`);
                setSummary(summaryRes.data?.summary_text || null);
            } catch (error) {
                console.error("Failed to load summary", error);
            }

            // Fetch action items with full data
            const actionsRes = await client.get(`/meetings/${id}/action-items/`);
            setActions(actionsRes.data);
        } catch (error) {
            console.error("Failed to load meeting details", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (id) {
            fetchMeetingInfo();
        }
    }, [id]);

    useEffect(() => {
        if (isComplete) {
            fetchMeetingDetails();
        }
    }, [isComplete, id]);

    // While waiting for completion
    if (!isComplete) {
        return (
            <div className="w-full max-w-7xl mx-auto px-4 py-8">
                <ProgressPanel meetingId={Number(id)} onComplete={() => setIsComplete(true)} />
                {loading && (
                    <div className="mt-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 text-center">
                        <p className="text-white">Loading meeting details...</p>
                    </div>
                )}
            </div>
        );
    }

    // After the completion of background ingestion task
    return (
        <div className="w-full max-w-7xl mx-auto space-y-6 px-4 py-8">
            {/* Meeting Header */}
            {meetingInfo && (
                <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-6 shadow-xl">
                    <h1 className="text-3xl font-bold text-white mb-2">{meetingInfo.title || meetingInfo.filename}</h1>
                    <div className="flex items-center space-x-4 text-white/70">
                        <span className="flex items-center space-x-2">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span>{meetingInfo.filename}</span>
                        </span>
                        <span className="flex items-center space-x-2">
                            <div className={`w-2 h-2 rounded-full ${meetingInfo.status === 'completed' ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                            <span className="capitalize">{meetingInfo.status}</span>
                        </span>
                    </div>
                </div>
            )}

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Transcript */}
                <div className="lg:col-span-2 space-y-6">
                    <TranscriptPanel transcript={transcript} />
                    <SearchPanel meetingId={Number(id)} />
                </div>

                {/* Right Column - Summary and Action Items */}
                <div className="space-y-6">
                    <SummaryPanel summary={summary} />
                    <ActionItemsPanel actions={actions} />
                </div>
            </div>
        </div>
    );
}
