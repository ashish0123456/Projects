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
        <div className="space-y-4">
            <UploadForm onUploaded={handleUploaded} />
            {meetingId && <p>We’re processing your meeting. Redirecting to the meeting page…</p>}
        </div>
    );
}