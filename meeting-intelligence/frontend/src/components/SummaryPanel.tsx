interface Props {
    summary: string;
}

export default function SummaryPanel({ summary }: Props) {
    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="font-semibold mb-2">Meeting Summary</h2>
            <p className="text-sm">{summary}</p>
        </div>
    );
}