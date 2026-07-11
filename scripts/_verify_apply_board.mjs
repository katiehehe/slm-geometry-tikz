/**
 * Headless Chromium proof: Apply board edits → figure / TikZ / chat state.
 * Run from /tmp (where playwright is installed):
 *   node /tmp/verify_apply_board.mjs
 */
import { chromium } from "playwright";
import fs from "fs";

const URL = "https://katie-he--geotikz-copilot-web.modal.run";
const USER = "demo";
const PASS = "geotikz-gpu-8t3n";
const OUT = "/Users/katiehe/dev/projects/slm-geometry-tikz/outputs/verify_apply_board";
fs.mkdirSync(OUT, { recursive: true });

const CIRCUM = String.raw`\begin{tikzpicture}
\tkzDefPoint(0,0){A}\tkzDefPoint(6,0){B}\tkzDefPoint(1,4){C}
\tkzDefTriangleCenter[circum](A,B,C)\tkzGetPoint{O}
\tkzDrawPolygon(A,B,C)\tkzDrawCircle(O,A)
\tkzDrawPoints(A,B,C,O)\tkzLabelPoints(A,B,C,O)
\end{tikzpicture}`;

const log = (...a) => console.log(...a);

async function boardFrame(page) {
  const fr = page.frames().find((f) => (f.name() || "").includes("") && f.url().startsWith("about:srcdoc"));
  // Prefer title match via element
  const handle = await page.$('iframe[title="interactive geometry editor"]');
  if (!handle) return null;
  return await handle.contentFrame();
}

async function main() {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const context = await browser.newContext({
    httpCredentials: { username: USER, password: PASS },
    viewport: { width: 1400, height: 1200 },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(120_000);

  log("→ open", URL);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 180_000 });
  await page.waitForSelector("#msg-box", { timeout: 60_000 });
  await page.waitForTimeout(1500);
  log("booted");

  // Paste TikZ (deterministic)
  await page.getByText("More input types").first().click();
  await page.waitForTimeout(400);
  await page.getByLabel(/Paste a full tikzpicture/i).fill(CIRCUM);
  await page.getByRole("button", { name: /Render & edit this TikZ/i }).click();
  log("→ render pasted TikZ…");

  await page.waitForSelector('iframe[title="interactive geometry editor"]', { timeout: 120_000 });
  log("iframe present");

  let frame = null;
  for (let i = 0; i < 60; i++) {
    frame = await boardFrame(page);
    if (frame) {
      const ready = await frame.evaluate(() => !!(window.__toTikz && window.__P && window.__P.A && window.__P.O));
      if (ready) break;
      frame = null;
    }
    await page.waitForTimeout(1000);
  }
  if (!frame) {
    await page.screenshot({ path: `${OUT}/fail_no_board.png`, fullPage: true });
    throw new Error("board frame never booted");
  }
  log("board ready");

  const beforeCoords = await frame.evaluate(() => {
    const xy = (p) => [+p.X().toFixed(3), +p.Y().toFixed(3)];
    return { A: xy(__P.A), B: xy(__P.B), C: xy(__P.C), O: xy(__P.O) };
  });
  log("before", beforeCoords);

  const codeBefore = await page.evaluate(() => {
    const cm = document.querySelector(".cm-content");
    return cm ? cm.innerText : "";
  });

  await page.screenshot({ path: `${OUT}/01_before_apply.png`, fullPage: true });
  log("shot 01_before_apply.png");

  // Move free point A; constrained O should re-solve
  await frame.evaluate(() => {
    __P.A.setPosition(JXG.COORDS_BY_USER, [1.5, 0.8]);
    __board.update();
  });
  await page.waitForTimeout(600);

  const afterDrag = await frame.evaluate(() => {
    const xy = (p) => [+p.X().toFixed(3), +p.Y().toFixed(3)];
    return { A: xy(__P.A), B: xy(__P.B), C: xy(__P.C), O: xy(__P.O), tikz: __toTikz() };
  });
  log("after drag", { A: afterDrag.A, O: afterDrag.O });

  const aMoved = Math.hypot(afterDrag.A[0] - beforeCoords.A[0], afterDrag.A[1] - beforeCoords.A[1]) > 0.05;
  const oMoved = Math.hypot(afterDrag.O[0] - beforeCoords.O[0], afterDrag.O[1] - beforeCoords.O[1]) > 0.05;

  // Apply via Gradio button (JS bridge pulls TikZ via postMessage)
  log("→ Apply board edits");
  await page.locator("#apply-board-btn button, #apply-board-btn").first().click();
  await page.waitForFunction(
    () => /applied board edits|Applied board edits/i.test(document.body.innerText),
    null,
    { timeout: 120_000 },
  );
  await page.waitForTimeout(1200);

  const codeAfter = await page.evaluate(() => {
    const cm = document.querySelector(".cm-content");
    return cm ? cm.innerText : "";
  });
  const body = await page.evaluate(() => document.body.innerText);
  const appliedNote = /applied board edits/i.test(body);
  const hasNewA = /\\coordinate\s*\(\s*A\s*\)\s*at\s*\(\s*1\.5/.test(codeAfter);

  const plainErrorLabels = await page.evaluate(() =>
    [...document.querySelectorAll("*")].filter((el) => el.childNodes.length === 1 && el.textContent.trim() === "Error").length,
  );

  await page.screenshot({ path: `${OUT}/02_after_apply.png`, fullPage: true });
  log("shot 02_after_apply.png");

  // Follow-up: chat edit should see new cur_tikz — send a tiny label tweak and ensure no crash.
  // Use a fast clarify-safe edit; if frontier is slow, still count Apply success above.
  let followUpOk = null;
  try {
    await page.locator("#msg-box textarea, #msg-box").first().fill("make the labels slightly larger");
    await page.locator("#send-btn button, #send-btn").first().click();
    await page.waitForFunction(
      () => !document.body.innerText.includes("…drawing…") || /edited by|CLARIFY|hiccuped|Applied/i.test(document.body.innerText),
      null,
      { timeout: 90_000 },
    );
    await page.waitForTimeout(1000);
    followUpOk = !/Error/.test(await page.locator("#fig-out").innerText().catch(() => ""));
    await page.screenshot({ path: `${OUT}/03_after_chat_edit.png`, fullPage: true });
  } catch (e) {
    followUpOk = false;
    log("follow-up edit note:", String(e).slice(0, 200));
  }

  const summary = {
    aMoved,
    oMoved,
    appliedNote,
    hasNewA,
    plainErrorLabels,
    codeBeforeSnippet: codeBefore.slice(0, 160),
    codeAfterSnippet: codeAfter.slice(0, 240),
    beforeCoords,
    afterDrag: { A: afterDrag.A, O: afterDrag.O },
    followUpOk,
  };
  fs.writeFileSync(`${OUT}/result.json`, JSON.stringify(summary, null, 2));
  log("RESULT", JSON.stringify(summary, null, 2));

  const ok = aMoved && oMoved && appliedNote && hasNewA && plainErrorLabels === 0;
  if (!ok) {
    console.error("VERIFY FAILED");
    process.exitCode = 1;
  } else {
    log("VERIFY PASSED");
  }
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
