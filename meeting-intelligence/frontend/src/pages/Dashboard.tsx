import { useState } from "react";
import UploadForm from "../components/UploadForm";
import ProgressPanel from "../components/ProgressPanel";
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
            {meetingId && <ProgressPanel meetingId={meetingId} />}
        </div>
    );
}