import Shell from "./layout/Shell";
import Home from "./pages/Home";
import Workspace from "./pages/Workspace";
import Feed from "./pages/Feed";
import Harness from "./pages/Harness";

export default function App() {
  return (
    <Shell>
      {(route, state) => {
        if (route.page === "home") return <Home {...state} />;
        if (route.page === "feed") return <Feed {...state} />;
        if (route.page === "harness") return <Harness sessionId={state.sessionId} />;
        return <Workspace {...state} />;
      }}
    </Shell>
  );
}
