import { useState, useRef } from "react";
import client from "../api/client";

interface Props {
    onUploaded: (meetingId: Number) => void;
}

export default function UploadForm({ onUploaded }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

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

    const formatFileSize = (bytes: number) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    };

    return (
        <div className="max-w-2xl mx-auto">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl shadow-2xl border border-white/20 p-8">
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-white mb-2">Upload Meeting Recording</h2>
                    <p className="text-white/70">Upload your audio or video file to get AI-powered insights</p>
                </div>

                <div
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${dragActive
                            ? "border-purple-400 bg-purple-500/20"
                            : "border-white/30 hover:border-white/50 bg-white/5"
                        }`}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".mp4,.mov,.avi,audio/*,.mp3,.wav,.m4a"
                        onChange={handleFileChange}
                        className="hidden"
                    />

                    <div className="flex flex-col items-center space-y-4">
                        <div className="w-20 h-20 bg-gradient-to-br from-purple-400 to-pink-400 rounded-full flex items-center justify-center shadow-lg">
                            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                        </div>

                        {file ? (
                            <div className="space-y-2">
                                <p className="text-white font-semibold">{file.name}</p>
                                <p className="text-white/60 text-sm">{formatFileSize(file.size)}</p>
                                <button
                                    onClick={() => setFile(null)}
                                    className="text-purple-300 hover:text-purple-200 text-sm underline"
                                >
                                    Remove file
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <p className="text-white font-medium">
                                    Drag and drop your file here, or
                                </p>
                                <button
                                    onClick={() => fileInputRef.current?.click()}
                                    className="text-purple-300 hover:text-purple-200 underline"
                                >
                                    browse to upload
                                </button>
                                <p className="text-white/50 text-sm mt-4">
                                    Supports: MP4, MOV, AVI, MP3, WAV, M4A
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {file && (
                    <button
                        onClick={handleUpload}
                        disabled={uploading}
                        className="mt-6 w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold py-4 px-6 rounded-xl hover:from-purple-600 hover:to-pink-600 transform hover:scale-[1.02] transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                    >
                        {uploading ? (
                            <span className="flex items-center justify-center space-x-2">
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <span>Processing...</span>
                            </span>
                        ) : (
                            "Upload & Process Meeting"
                        )}
                    </button>
                )}
            </div>
        </div>
    );
}