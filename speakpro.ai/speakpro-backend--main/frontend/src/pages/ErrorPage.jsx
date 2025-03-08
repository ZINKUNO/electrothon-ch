import React from "react";
import { ReactComponent as ErrorImage } from "../assets/error404.svg";
import errorStyles from "../styles/error.module.css";

export default function ErrorPage() {
  return (
    <div className={errorStyles.errorPage}>
      <p>
        <i>P.S. Click on logo to go home</i>
      </p>
      <div className={errorStyles.errorImage}>
        <ErrorImage />
      </div>
    </div>
  );
}
