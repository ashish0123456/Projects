import { useState } from "react";
import client from "../api/client";

interface Props {
    onUploaded: (meetingId: Number) => void;
}

export default function UploadForm({ onUploaded }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append("file", file);
        formData.append("title", file.name.replace(/\.[^/.]+$/, ""));

        setUploading(true);
        try {
            const res = await client.post("/meetings/upload/", formData, {
                headers: { "Content-Type": "multipart/form-data" }
            });
            onUploaded(res.data.id);
        } catch (error) {
            console.error("Upload failed", error);
            alert("Upload failed. Please try again.");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="p-4 bg-white rounted-lg shadow-md">
            <input
                type="file"
                accept=".mp4,.mov,.avi, audio/*"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="mb-3"
            />
            <button
                onClick={handleUpload}
                disabled={uploading}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
            >
                {uploading ? "Uploading..." : "Upload & Process"}
            </button>
        </div>
    );
}