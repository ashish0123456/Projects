import { useState } from "react";
import UploadForm from "../components/UploadForm";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
    const [meetingId, setMeetingId] = useState<Number | null>(null);
    const navigate = useNavigate();

    const handleUploaded = (id: Number) => {
        setMeetingId(id);
        setTimeout(() => {
            navigate(`/meeting/${id}`);
        }, 2000); // Navigate after 2 second
    }

    return (
        <div className="w-full max-w-7xl mx-auto space-y-6 px-4 py-8">
            <UploadForm onUploaded={handleUploaded} />
            {meetingId && (
                <div className="w-full max-w-2xl mx-auto bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 text-center">
                    <div className="flex items-center justify-center space-x-3">
                        <svg className="animate-spin h-6 w-6 text-purple-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <p className="text-white font-medium">We're processing your meeting. Redirecting to the meeting page…</p>
                    </div>
                </div>
            )}
        </div>
    );
}