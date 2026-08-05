// Playwright E2E Test Suite for AirOllama Agentic Coding Interface
const { test, expect } = require('@playwright/test');

const BASE_URL = 'http://127.0.0.1:11211';

test.describe('AirOllama Agentic Coding Interface', () => {

  test('should render agentic coding layout, sidebar, and controls', async ({ page }) => {
    // Navigate to AirOllama Dashboard
    await page.goto(`${BASE_URL}/dashboard`);

    // Click Agentic Coding navbar tab
    const agenticTab = page.locator('button.tab-btn:has-text("Agentic Coding")');
    await expect(agenticTab).toBeVisible({ timeout: 10000 });
    await agenticTab.click();

    // Wait for modular view template to load
    const promptInput = page.locator('#agentic-prompt-input');
    await expect(promptInput).toBeVisible({ timeout: 10000 });

    // Verify sidebar components
    await expect(page.locator('button:has-text("New Conversation")')).toBeVisible();
    await expect(page.locator('#agentic-conv-list')).toBeVisible();
    await expect(page.locator('#agentic-project-list')).toBeVisible();

    // Verify header title & project tag
    await expect(page.locator('#agentic-active-title')).toBeVisible();
    await expect(page.locator('#agentic-active-project-tag')).toBeVisible();

    // Verify input bar controls
    await expect(page.locator('#agentic-model-select')).toBeVisible();
    await expect(page.locator('#agentic-mode-select')).toBeVisible();
    await expect(page.locator('#agentic-btn-send')).toBeVisible();
  });

  test('should submit prompt and render assistant response stream', async ({ page }) => {
    await page.goto(`${BASE_URL}/dashboard`);

    // Open Agentic Coding view
    await page.click('button.tab-btn:has-text("Agentic Coding")');
    const promptInput = page.locator('#agentic-prompt-input');
    await expect(promptInput).toBeVisible({ timeout: 10000 });

    // Type prompt into input textarea
    await promptInput.fill('Write a short Python function to calculate Fibonacci numbers.');

    // Click Send button
    await page.click('#agentic-btn-send');

    // Verify assistant markdown body rendered
    const agentResponse = page.locator('.agentic-md-body');
    await expect(agentResponse).toBeVisible({ timeout: 15000 });
  });

});
