interface Props {
    transcript: string[];
}

export default function TranscriptPanel({ transcript }: Props) {
    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="font-semibold mb-2">Transcript</h2>
            <div className="space-y-2 text-sm">
                {transcript.map((line, index) => (
                    <p key={index} className="border-b pb-1">{line}</p>
                ))}
            </div>
        </div>
    );
}