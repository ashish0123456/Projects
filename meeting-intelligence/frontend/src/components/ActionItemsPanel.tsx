interface ActionItem {
    id: number;
    meeting_id: number;
    segment_id: number | null;
    text: string;
    assigned_to: string | null;
    resolved: boolean;
}

interface Props {
    actions: ActionItem[];
}

export default function ActionItemsPanel({ actions }: Props) {
    if (actions.length === 0) {
        return (
            <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
                <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-orange-400 to-red-400 rounded-lg flex items-center justify-center">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-bold text-white">Action Items</h2>
                </div>
                <p className="text-white/60">No action items found.</p>
            </div>
        );
    }

    const resolvedCount = actions.filter(a => a.resolved).length;
    const totalCount = actions.length;

    return (
        <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-orange-400 to-red-400 rounded-lg flex items-center justify-center">
                        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-bold text-white">Action Items</h2>
                </div>
                <span className="text-white/60 text-sm">
                    {resolvedCount}/{totalCount} completed
                </span>
            </div>
            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                {actions.map((action) => (
                    <div 
                        key={action.id}
                        className={`bg-white/5 rounded-lg p-4 border transition-all ${
                            action.resolved 
                                ? 'border-green-500/30 bg-green-500/5 opacity-75' 
                                : 'border-white/10 hover:bg-white/10'
                        }`}
                    >
                        <div className="flex items-start space-x-3">
                            <div className="flex-shrink-0 mt-1">
                                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                                    action.resolved 
                                        ? 'bg-green-500 border-green-500' 
                                        : 'border-white/40'
                                }`}>
                                    {action.resolved && (
                                        <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                        </svg>
                                    )}
                                </div>
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className={`text-white/90 leading-relaxed ${
                                    action.resolved ? 'line-through' : ''
                                }`}>
                                    {action.text}
                                </p>
                                {action.assigned_to && (
                                    <div className="mt-2 flex items-center space-x-2">
                                        <svg className="w-4 h-4 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                        </svg>
                                        <span className="text-white/60 text-xs">{action.assigned_to}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}