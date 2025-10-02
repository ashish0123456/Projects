interface Props {
    actions: string[];
}

export default function ActionItemsPanel({ actions }: Props) {
    return (
        <div className="p-4 bg-white rounded-lg shadow-md">
            <h2 className="font-semibold mb-2">Action Items</h2>
            <ul className="list-disc list-inside text-sm space-y-1">
                {actions.map((action, index) => (
                    <li key={index}>{action}</li>
                ))}
            </ul>
        </div>
    );
}