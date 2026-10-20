import { HarnessSessionViewer } from "../components/harness-viewer";

export default function Harness({ sessionId }: { sessionId: string | null }) {
  return (
    <div className="page-scroll">
      <div className="glass">
        <HarnessSessionViewer sessionId={sessionId} />
      </div>
    </div>
  );
}
