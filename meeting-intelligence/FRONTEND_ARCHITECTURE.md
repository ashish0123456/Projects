# Frontend Architecture - Complete Flow Explanation

## 📋 Table of Contents
1. [Application Entry Point](#1-application-entry-point)
2. [App Component & Routing](#2-app-component--routing)
3. [Navigation Flow](#3-navigation-flow)
4. [Component Hierarchy](#4-component-hierarchy)
5. [Data Flow & API Communication](#5-data-flow--api-communication)
6. [Complete User Journey](#6-complete-user-journey)

---

## 1. Application Entry Point

### `main.tsx` - The Bootstrap

```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
```

**What happens here:**
- **`createRoot()`**: React 18's new root API - creates the React root and attaches it to the DOM element with id `root`
- **`StrictMode`**: Development tool that helps identify potential problems (double-renders, deprecated APIs, etc.)
- **`BrowserRouter`**: Enables client-side routing using HTML5 History API (allows `/meeting/123` URLs without page reloads)
- **`<App />`**: The main application component

**Flow:** Browser loads → React mounts → Router initializes → App component renders

---

## 2. App Component & Routing

### `App.tsx` - The Root Layout

```typescript
export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br...">
      <Navbar />
      <div className="relative container mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/meeting/:id" element={<MeetingView />} />
        </Routes>
      </div>
    </div>
  );
}
```

**Component Structure:**
```
App
├── Background (gradient + pattern overlay)
├── Navbar (always visible)
└── Routes Container
    ├── Route "/" → Dashboard
    └── Route "/meeting/:id" → MeetingView
```

**Key Concepts:**
- **`Routes`**: Container that matches current URL to route definitions
- **`Route path="/"`**: Matches when URL is exactly `/`
- **`Route path="/meeting/:id"`**: Dynamic route - `:id` is a URL parameter (e.g., `/meeting/5` → `id = "5"`)
- **`element={<Component />}`**: Component to render when route matches

**Navigation happens via:**
1. User clicks `<Link>` components (declarative)
2. Programmatic navigation using `useNavigate()` hook (imperative)

---

## 3. Navigation Flow

### How Navigation Works

#### **Declarative Navigation (Links)**
```typescript
// In Navbar.tsx
<Link to="/">Dashboard</Link>
```
- User clicks → React Router intercepts → URL changes → Route matches → Component renders
- **No page reload** - Single Page Application (SPA) behavior

#### **Programmatic Navigation (Hooks)**
```typescript
// In Dashboard.tsx
const navigate = useNavigate();
navigate(`/meeting/${id}`);
```
- Function call → URL changes → Route matches → Component renders

### Navigation States

**State 1: Dashboard (`/`)**
```
URL: /
Active Route: <Route path="/" element={<Dashboard />} />
Rendered: Dashboard → UploadForm
```

**State 2: Meeting View (`/meeting/123`)**
```
URL: /meeting/123
Active Route: <Route path="/meeting/:id" element={<MeetingView />} />
URL Params: { id: "123" }
Rendered: MeetingView → ProgressPanel (if processing) OR MeetingView → All panels (if complete)
```

---

## 4. Component Hierarchy

### Complete Component Tree

```
App
│
├── Navbar (presentational)
│   ├── Logo Link (to="/")
│   └── Dashboard Link (to="/")
│
└── Routes Container
    │
    ├── Route "/" → Dashboard
    │   ├── UploadForm
    │   │   ├── File Input (hidden)
    │   │   ├── Drag & Drop Zone
    │   │   └── Upload Button
    │   └── Loading Message (conditional)
    │
    └── Route "/meeting/:id" → MeetingView
        │
        ├── [If !isComplete] ProgressPanel
        │   └── Polls API every 2s → Calls onComplete() when done
        │
        └── [If isComplete] Full Meeting View
            ├── Meeting Header (meetingInfo)
            └── Content Grid
                ├── Left Column (2/3 width)
                │   ├── TranscriptPanel (transcript[])
                │   └── SearchPanel (meetingId)
                │
                └── Right Column (1/3 width)
                    ├── SummaryPanel (summary)
                    └── ActionItemsPanel (actions[])
```

---

## 5. Data Flow & API Communication

### API Client Setup

**`api/client.ts`**
```typescript
const client = axios.create({
  baseURL: '/api/v1',  // All requests go to /api/v1/*
  headers: { 'Content-Type': 'application/json' },
  withCredentials: true,
});
```

**How it works:**
- Vite proxy (in `vite.config.ts`) forwards `/api/*` → `http://localhost:8000`
- All components import `client` and make requests like `client.get('/meetings/123/')`
- Actual request: `GET http://localhost:8000/api/v1/meetings/123/`

### Data Flow Patterns

#### **Pattern 1: Parent → Child (Props Down)**
```
Dashboard (state: meetingId)
    ↓ passes callback
UploadForm (receives: onUploaded)
    ↓ calls callback with data
Dashboard (receives: id)
    ↓ updates state
Dashboard (navigates to /meeting/:id)
```

#### **Pattern 2: Child → Parent (Callback Up)**
```
MeetingView (state: isComplete)
    ↓ passes callback
ProgressPanel (receives: onComplete)
    ↓ detects completion
ProgressPanel (calls: onComplete())
    ↓ triggers state update
MeetingView (sets: isComplete = true)
    ↓ re-renders
MeetingView (shows: Full meeting view)
```

#### **Pattern 3: Component → API → State**
```
Component mounts
    ↓ useEffect triggers
Component calls: client.get('/api/endpoint')
    ↓ axios makes HTTP request
Backend responds with data
    ↓ response.data
Component calls: setState(data)
    ↓ React re-renders
Component displays new data
```

---

## 6. Complete User Journey

### Journey 1: Uploading a Meeting

**Step 1: User lands on Dashboard**
```
1. Browser navigates to: http://localhost:5173/
2. React Router matches: Route path="/" → Dashboard
3. Dashboard renders → UploadForm renders
4. User sees: Upload form with drag & drop zone
```

**Step 2: User selects file**
```
1. User drags file OR clicks "browse"
2. UploadForm.handleFileChange() OR handleDrop() fires
3. setFile(file) updates state
4. Component re-renders → Shows file name and size
5. "Upload & Process" button appears
```

**Step 3: User clicks upload**
```
1. User clicks "Upload & Process"
2. handleUpload() executes:
   a. Creates FormData with file + title
   b. Sets uploading = true (shows spinner)
   c. Calls: client.post("/meetings/upload/", formData)
   d. Backend responds: { id: 123, title: "...", ... }
   e. Calls: onUploaded(res.data.id) ← Callback to parent
3. Dashboard.handleUploaded(123) executes:
   a. setMeetingId(123)
   b. Shows "Processing..." message
   c. setTimeout(() => navigate(`/meeting/123`), 2000)
4. After 2 seconds → Navigation happens
```

**Step 4: Navigation to Meeting View**
```
1. URL changes to: /meeting/123
2. React Router matches: Route path="/meeting/:id"
3. MeetingView component mounts
4. useParams() extracts: { id: "123" }
5. isComplete = false (initial state)
6. Renders: ProgressPanel only
```

### Journey 2: Processing & Viewing Results

**Step 1: Progress Monitoring**
```
1. ProgressPanel mounts with meetingId={123}
2. useEffect runs:
   a. Calls fetchProgress() immediately
   b. Sets interval to poll every 2 seconds
   c. API: GET /meetings/123/progress/
   d. Response: { stage: "transcribing", detail: { msg: "..." } }
   e. setProgress(data) → Shows current stage
3. Every 2 seconds: Polls again
4. When stage === "completed":
   a. setIsComplete(true)
   b. Calls onComplete() ← Callback to MeetingView
   c. Clears interval
```

**Step 2: MeetingView receives completion**
```
1. MeetingView.onComplete() callback fires
2. setIsComplete(true) updates state
3. Component re-renders
4. Condition: if (isComplete) → Shows full view
5. useEffect triggers (dependency: isComplete)
6. fetchMeetingDetails() executes:
   a. GET /meetings/123/transcripts/ → setTranscript(data)
   b. GET /meetings/123/summary/ → setSummary(data)
   c. GET /meetings/123/action-items/ → setActions(data)
7. All panels receive their data via props
```

**Step 3: Displaying Results**
```
MeetingView renders:
├── Meeting Header
│   └── Shows: title, filename, status
│
├── TranscriptPanel
│   ├── Receives: transcript[] (array of segments)
│   ├── Maps over segments
│   └── Displays: speaker, timestamps, text
│
├── SummaryPanel
│   ├── Receives: summary (string)
│   └── Displays: formatted summary text
│
├── ActionItemsPanel
│   ├── Receives: actions[] (array of action items)
│   ├── Maps over actions
│   └── Displays: checkbox, text, assigned_to
│
└── SearchPanel
    ├── Receives: meetingId (number)
    ├── User types query
    ├── Calls: POST /meetings/123/search/ { query, top_k: 5 }
    └── Displays: search results with scores
```

---

## 7. Component Communication Patterns

### Pattern A: Props (Parent → Child)
```typescript
// Parent passes data down
<TranscriptPanel transcript={transcript} />

// Child receives and uses
function TranscriptPanel({ transcript }: Props) {
  return transcript.map(segment => ...)
}
```

### Pattern B: Callbacks (Child → Parent)
```typescript
// Parent passes callback
<ProgressPanel onComplete={() => setIsComplete(true)} />

// Child calls callback when ready
if (data?.stage === "completed") {
  onComplete(); // ← Notifies parent
}
```

### Pattern C: Hooks (Component → Router/API)
```typescript
// useParams - Get URL parameters
const { id } = useParams(); // { id: "123" }

// useNavigate - Programmatic navigation
const navigate = useNavigate();
navigate('/meeting/123');

// useState - Local component state
const [data, setData] = useState(null);

// useEffect - Side effects (API calls, subscriptions)
useEffect(() => {
  fetchData();
  return () => cleanup(); // Cleanup on unmount
}, [dependencies]);
```

### Pattern D: API Client (Component → Backend)
```typescript
// All components use the same client
import client from '../api/client';

// GET request
const res = await client.get('/meetings/123/');
const data = res.data;

// POST request
const res = await client.post('/meetings/123/search/', {
  query: 'keyword',
  top_k: 5
});
```

---

## 8. State Management Flow

### State Locations

**1. Local Component State (useState)**
- `UploadForm`: `file`, `uploading`, `dragActive`
- `Dashboard`: `meetingId`
- `MeetingView`: `meetingInfo`, `transcript`, `summary`, `actions`, `loading`, `isComplete`
- `ProgressPanel`: `progress`, `isComplete`
- `SearchPanel`: `query`, `results`, `searching`

**2. URL State (React Router)**
- Current route: `/` or `/meeting/:id`
- URL parameters: `{ id: "123" }` from `useParams()`

**3. Server State (API Responses)**
- Fetched on-demand
- Stored in component state
- Refreshed via API calls

### State Updates Trigger Re-renders

```
User Action
    ↓
Event Handler
    ↓
setState(newValue)
    ↓
React detects change
    ↓
Component re-renders
    ↓
Child components receive new props
    ↓
UI updates
```

---

## 9. Key React Concepts Used

### 1. **Component Composition**
- Small, focused components combined to build complex UIs
- Example: `MeetingView` composes `TranscriptPanel`, `SummaryPanel`, etc.

### 2. **Conditional Rendering**
```typescript
{isComplete ? (
  <FullView />
) : (
  <ProgressPanel />
)}
```

### 3. **Lists & Keys**
```typescript
{transcript.map((segment) => (
  <div key={segment.id}>...</div>
))}
```

### 4. **Event Handling**
```typescript
onClick={handleUpload}
onChange={(e) => setFile(e.target.files?.[0])}
onDragEnter={handleDrag}
```

### 5. **Effect Hooks**
```typescript
useEffect(() => {
  // Runs after render
  fetchData();
  
  return () => {
    // Cleanup (runs on unmount)
    clearInterval(interval);
  };
}, [dependencies]); // Re-runs when dependencies change
```

### 6. **Refs (DOM Access)**
```typescript
const fileInputRef = useRef<HTMLInputElement>(null);
fileInputRef.current?.click(); // Programmatically click hidden input
```

---

## 10. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    REACT COMPONENTS                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Dashboard│  │MeetingView│  │UploadForm│  │Progress  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│       │             │              │              │          │
│       ▼             ▼              ▼              ▼          │
│  ┌────────────────────────────────────────────────────┐    │
│  │              API CLIENT (axios)                    │    │
│  │  baseURL: '/api/v1'                                │    │
│  └────────────────────┬───────────────────────────────┘    │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    VITE PROXY                                │
│  /api/* → http://localhost:8000                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND API                               │
│  FastAPI Endpoints:                                          │
│  - POST /meetings/upload/                                    │
│  - GET  /meetings/{id}/                                      │
│  - GET  /meetings/{id}/progress/                             │
│  - GET  /meetings/{id}/transcripts/                          │
│  - GET  /meetings/{id}/summary/                              │
│  - GET  /meetings/{id}/action-items/                         │
│  - POST /meetings/{id}/search/                               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATABASE                                  │
│  PostgreSQL: meetings, transcripts, summaries, action_items   │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Real-World Example Walkthrough

### Scenario: User uploads "team-meeting.mp4"

**Timeline:**

**T=0s: User clicks "Upload & Process"**
- `UploadForm.handleUpload()` executes
- `client.post("/meetings/upload/", formData)` → Backend
- Backend: Creates meeting record, queues Celery task
- Response: `{ id: 42, title: "team-meeting", status: "uploaded" }`
- `onUploaded(42)` → Dashboard
- Dashboard: `setMeetingId(42)`, shows "Processing..." message

**T=2s: Auto-navigation**
- `setTimeout` fires → `navigate('/meeting/42')`
- URL changes → React Router matches route
- `MeetingView` mounts with `id = "42"`
- `isComplete = false` → Shows `ProgressPanel`

**T=2s-30s: Progress polling**
- `ProgressPanel` polls every 2s: `GET /meetings/42/progress/`
- Stages: `uploaded` → `transcribing` → `diarizing` → `summarizing` → `extracting`
- UI updates with current stage icon and message

**T=30s: Processing completes**
- Backend: `stage = "completed"`
- `ProgressPanel` detects completion
- Calls `onComplete()` → `MeetingView.setIsComplete(true)`
- `MeetingView` re-renders → Shows full view

**T=30s: Data fetching**
- `useEffect` triggers `fetchMeetingDetails()`
- Parallel API calls:
  - `GET /meetings/42/transcripts/` → 150 segments
  - `GET /meetings/42/summary/` → "The team discussed..."
  - `GET /meetings/42/action-items/` → 8 action items
- State updates → All panels receive data

**T=31s: User views results**
- `TranscriptPanel`: Shows 150 segments with speakers and timestamps
- `SummaryPanel`: Shows AI-generated summary
- `ActionItemsPanel`: Shows 8 tasks with completion status
- `SearchPanel`: Ready for user queries

**T=35s: User searches**
- User types: "budget"
- Clicks "Search" → `POST /meetings/42/search/ { query: "budget", top_k: 5 }`
- Backend: Vector search returns 5 matching segments
- `SearchPanel` displays results with relevance scores

---

## 12. Key Takeaways

1. **Single Page Application (SPA)**: No page reloads - React Router handles navigation
2. **Component Hierarchy**: Parent components pass data down, children notify parents via callbacks
3. **State Management**: Each component manages its own state, shared via props
4. **API Communication**: Centralized `client` instance, all requests go through it
5. **Reactive Updates**: State changes trigger automatic re-renders
6. **Lifecycle Management**: `useEffect` handles side effects (API calls, subscriptions)
7. **URL as State**: React Router syncs URL with component state

---

## 13. Code Correlation Map

| Component | Key Responsibilities | State | API Calls | Props Received | Callbacks |
|-----------|---------------------|-------|-----------|-----------------|-----------|
| **App** | Layout, Routing | None | None | None | None |
| **Navbar** | Navigation UI | None | None | None | None |
| **Dashboard** | Upload flow | `meetingId` | None | None | `onUploaded` |
| **UploadForm** | File upload | `file`, `uploading` | `POST /upload/` | `onUploaded` | Calls `onUploaded(id)` |
| **MeetingView** | Orchestrates meeting display | `meetingInfo`, `transcript`, `summary`, `actions`, `isComplete` | Multiple GETs | None | `onComplete` |
| **ProgressPanel** | Progress monitoring | `progress` | `GET /progress/` | `meetingId`, `onComplete` | Calls `onComplete()` |
| **TranscriptPanel** | Display transcript | None | None | `transcript[]` | None |
| **SummaryPanel** | Display summary | None | None | `summary` | None |
| **ActionItemsPanel** | Display action items | None | None | `actions[]` | None |
| **SearchPanel** | Search functionality | `query`, `results` | `POST /search/` | `meetingId` | None |

---

This architecture follows React best practices:
- ✅ Separation of concerns
- ✅ Reusable components
- ✅ Unidirectional data flow
- ✅ Clear component boundaries
- ✅ Proper state management
- ✅ Efficient re-rendering

