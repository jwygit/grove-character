/* Optional real-browser acceptance. No browser/mock API runs in production.
 * npm install --no-save playwright; npx playwright install chromium
 * node tests/workbench_browser.cjs http://127.0.0.1:8088 /absolute/video.mp4 /absolute/output
 * PLAYWRIGHT_MODULE / CHROME_EXECUTABLE can select an installed browser runtime.
 */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

(async () => {
  const [base, video, output] = process.argv.slice(2);
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_EXECUTABLE
      ? { executablePath: process.env.CHROME_EXECUTABLE }
      : {}),
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    await page.goto(base);
    await page.waitForSelector("#profile option", { state: "attached" });
    await page.locator("#file").setInputFiles(video);
    await page.locator("#upload").click();
    await page.waitForFunction(
      () => document.querySelector("#run-state").textContent === "已完成",
      null,
      { timeout: 60000 },
    );
    await page.waitForFunction(
      () => document.querySelectorAll(".event").length > 20,
    );
    await page.locator("#mode").selectOption("latest");
    await page.waitForFunction(
      () => document.querySelectorAll(".episode").length === 3,
    );
    await page.locator("#question").fill("red cup");
    await page.locator("#ask").click();
    await page.waitForSelector(".citation");
    assert.equal(
      await page.locator(".answer").first().getAttribute("data-version"),
      "3",
    );
    await page.locator(".citation").first().click();
    await page.waitForFunction(() => document.querySelector("#detail").open);
    assert.match(
      await page.locator("#detail-text").textContent(),
      /citation_id/,
    );
    await page.locator("#close-detail").click();
    await page.locator("#mode").selectOption("playback");
    await page.locator("#video").evaluate((video) => {
      video.currentTime = 5;
    });
    await page.waitForFunction(() =>
      document.querySelector("#snapshot-label").textContent.includes("尚无"),
    );
    assert.equal(
      await page.locator(".answer").first().getAttribute("data-version"),
      "3",
    );
    await page.locator("#video").evaluate((video) => {
      video.currentTime = 6.5;
    });
    await page.waitForFunction(
      () => document.querySelectorAll(".episode").length === 1,
    );
    await page.locator('[data-tab="memory"]').click();
    assert.ok((await page.locator("#memory-list .card").count()) > 0);
    await page.locator('[data-tab="stages"]').click();
    await page.screenshot({
      path: path.join(output, "desktop.png"),
      fullPage: true,
    });
    await page.reload();
    await page.waitForSelector(".answer");
    assert.equal(
      await page.locator(".answer").first().getAttribute("data-version"),
      "3",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({
      path: path.join(output, "mobile.png"),
      fullPage: false,
    });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    );
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify({
        status: "passed",
        upload: true,
        worker: true,
        sse: true,
        history: true,
        citations: true,
        reload: true,
        mobile: true,
        screenshots: output,
        model_accuracy_verified: false,
      }),
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
