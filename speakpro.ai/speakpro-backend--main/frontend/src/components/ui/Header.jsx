import headerStyles from "../../styles/header.module.css";
import { ReactComponent as Logo } from "../../assets/verbalmatelogo.svg";
import { Link } from "react-router-dom";

export default function Header() {
  return (
    <header className={headerStyles.appHeaderBar}>
      <Link to="/" className={headerStyles.appHeader}>
        <Logo />
        <h1 className={headerStyles.appName}>SpeakPro Ai</h1>
      </Link>
    </header>
  );
}
