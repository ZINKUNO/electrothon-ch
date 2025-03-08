import loaderStyles from "../../styles/loader.module.css";

export default function Loader({ text }) {
  return (
    <p className={loaderStyles.loading}>
      {text}
      {"  "}
      <div className={loaderStyles.loader}></div>
    </p>
  );
}
