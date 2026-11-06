import { Bell, Clock3, Menu } from "lucide-react";
import "./Header.css";

export function Header({
  clock,
  activeAlerts,
  navigate,
  setMobileNav,
}) {
  return (
    <header className="topbar">
      <button
        className="icon-button mobile-menu"
        aria-label="Open navigation"
        onClick={() => setMobileNav(true)}
      >
        <Menu size={19} />
      </button>
      <div className="topbar-actions">
        <div className="top-clock">
          <Clock3 size={14} />
          <span>{clock.toLocaleTimeString("en-GB", { timeZone: "UTC", hour12: false })}</span>
          <small>UTC</small>
        </div>
        <button
          className="icon-button notifications"
          title="Open alert center"
          aria-label="Open alert center"
          onClick={() => navigate("Alert Center")}
        >
          <Bell size={17} />
          {activeAlerts > 0 && <i>{activeAlerts}</i>}
        </button>
      </div>
    </header>
  );
}
