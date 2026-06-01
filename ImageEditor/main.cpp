// ImageEditor - imgui + Win32 + DirectX 11 frontend.
// Two-panel image editor: load -> brush-stroke stylize -> save (png/jpg).

#define _CRT_SECURE_NO_WARNINGS

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>

#include <d3d11.h>
#include <tchar.h>
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cwctype>
#include <string>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "image.h"
#include "stylize.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "comdlg32.lib")

// ---------- D3D globals ----------
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;

static bool CreateDeviceD3D(HWND hWnd);
static void CleanupDeviceD3D();
static void CreateRenderTarget();
static void CleanupRenderTarget();
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ---------- GPU texture wrapper ----------
struct GpuTexture
{
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11Texture2D*          tex = nullptr;
    int w = 0, h = 0;

    void Release()
    {
        if (srv) { srv->Release(); srv = nullptr; }
        if (tex) { tex->Release(); tex = nullptr; }
        w = h = 0;
    }
    bool Upload(ID3D11Device* dev, const Image& img)
    {
        if (!img.valid()) { Release(); return false; }

        // Fast path: same dimensions -> just stream new pixels into the existing texture.
        if (tex && srv && w == img.w && h == img.h)
        {
            ID3D11DeviceContext* ctx = nullptr;
            dev->GetImmediateContext(&ctx);
            if (ctx)
            {
                ctx->UpdateSubresource(tex, 0, nullptr, img.rgba.data(), UINT(img.w) * 4u, 0);
                ctx->Release();
                return true;
            }
        }

        Release();

        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = img.w;
        desc.Height = img.h;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        D3D11_SUBRESOURCE_DATA sub = {};
        sub.pSysMem = img.rgba.data();
        sub.SysMemPitch = img.w * 4;

        if (FAILED(dev->CreateTexture2D(&desc, &sub, &tex))) return false;

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = desc.Format;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        if (FAILED(dev->CreateShaderResourceView(tex, &srvDesc, &srv)))
        {
            tex->Release(); tex = nullptr;
            return false;
        }
        w = img.w;
        h = img.h;
        return true;
    }
};

// ---------- File I/O (unicode-safe) ----------
static bool LoadImageFromPath(const wchar_t* path, Image& out)
{
    FILE* f = nullptr;
    if (_wfopen_s(&f, path, L"rb") != 0 || !f) return false;
    int w, h, n;
    unsigned char* data = stbi_load_from_file(f, &w, &h, &n, 4);
    fclose(f);
    if (!data) return false;
    out.w = w; out.h = h;
    out.rgba.assign(data, data + size_t(w) * size_t(h) * 4);
    stbi_image_free(data);
    return true;
}

static void StbiWriteToFile(void* ctx, void* data, int size)
{
    fwrite(data, 1, size_t(size), static_cast<FILE*>(ctx));
}

static bool SaveImageToPath(const wchar_t* path, const Image& img)
{
    if (!img.valid()) return false;
    std::wstring p(path), ext;
    if (auto dot = p.find_last_of(L'.'); dot != std::wstring::npos) ext = p.substr(dot);
    for (auto& c : ext) c = (wchar_t)std::towlower(c);

    FILE* f = nullptr;
    if (_wfopen_s(&f, path, L"wb") != 0 || !f) return false;
    bool ok = false;
    if (ext == L".png")
        ok = stbi_write_png_to_func(StbiWriteToFile, f, img.w, img.h, 4,
                                    img.rgba.data(), img.w * 4) != 0;
    else if (ext == L".jpg" || ext == L".jpeg")
        ok = stbi_write_jpg_to_func(StbiWriteToFile, f, img.w, img.h, 4,
                                    img.rgba.data(), 92) != 0;
    fclose(f);
    return ok;
}

static std::string WideToUtf8(const std::wstring& w)
{
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// ---------- File dialogs ----------
static bool OpenFileDialog(HWND owner, std::wstring& outPath)
{
    wchar_t buf[MAX_PATH] = L"";
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter =
        L"Images (*.png;*.jpg;*.jpeg;*.bmp;*.tga)\0*.png;*.jpg;*.jpeg;*.bmp;*.tga\0"
        L"All Files (*.*)\0*.*\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&ofn)) { outPath = buf; return true; }
    return false;
}

static bool SaveFileDialog(HWND owner, std::wstring& outPath)
{
    wchar_t buf[MAX_PATH] = L"";
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = L"PNG (*.png)\0*.png\0JPEG (*.jpg)\0*.jpg;*.jpeg\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    ofn.lpstrDefExt = L"png";
    if (GetSaveFileNameW(&ofn))
    {
        outPath = buf;
        if (outPath.find_last_of(L'.') == std::wstring::npos)
            outPath += (ofn.nFilterIndex == 2) ? L".jpg" : L".png";
        return true;
    }
    return false;
}

// ---------- UI helpers ----------
static void DrawImagePane(const char* title, const GpuTexture& tex, const char* emptyHint)
{
    ImGui::BeginChild(title, ImVec2(0, 0), ImGuiChildFlags_Borders);
    ImGui::TextUnformatted(title);
    ImGui::Separator();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (tex.srv && tex.w > 0 && tex.h > 0)
    {
        const float sx = avail.x / float(tex.w);
        const float sy = avail.y / float(tex.h);
        const float s  = (sx < sy) ? sx : sy;
        ImVec2 disp(float(tex.w) * s, float(tex.h) * s);
        ImVec2 cur = ImGui::GetCursorPos();
        ImGui::SetCursorPos(ImVec2(cur.x + (avail.x - disp.x) * 0.5f,
                                   cur.y + (avail.y - disp.y) * 0.5f));
        ImGui::Image((ImTextureID)(intptr_t)tex.srv, disp);
    }
    else
    {
        ImGui::TextDisabled("%s", emptyHint);
    }
    ImGui::EndChild();
}

static void DrawSettingsPanel(BrushStrokeParams& p)
{
    ImGui::TextUnformatted("Brush Stroke");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Gradient", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static const int kKernels[] = { 3, 5, 7, 9 };
        int kIdx = 0;
        for (int i = 0; i < 4; ++i) if (kKernels[i] == p.gradientKernel) { kIdx = i; break; }
        const char* kLabels[] = { "3x3", "5x5", "7x7", "9x9" };
        if (ImGui::Combo("Kernel", &kIdx, kLabels, 4))
            p.gradientKernel = kKernels[kIdx];
        ImGui::SliderFloat("Edge thresh", &p.gradientThreshold, 0.f, 100.f, "%.1f");
    }

    if (ImGui::CollapsingHeader("Multi-scale", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("Max brush R",  &p.maxBrushRadius, 1, 24);
        ImGui::SliderInt("Num scales",   &p.numScales, 1, 6);
        ImGui::SliderFloat("Error thresh", &p.errorThreshold, 0.f, 200.f, "%.1f");
    }

    if (ImGui::CollapsingHeader("Strokes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("Sampling",    &p.samplingRatio, 0.05f, 1.0f, "%.2f");
        ImGui::SliderInt("Length (max)",  &p.strokeLength, 2, 96);
        ImGui::SliderFloat("Curve smooth",&p.curvatureFilter, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Color tol",   &p.colorTolerance, 0.f, 200.f, "%.1f");
        ImGui::SliderFloat("Texture deg", &p.textureAngleDeg, 0.f, 359.f, "%.0f");
    }

    if (ImGui::CollapsingHeader("Compositing"))
    {
        ImGui::Checkbox("Blend (alpha)", &p.blend);
        ImGui::BeginDisabled(!p.blend);
        ImGui::SliderFloat("Alpha", &p.blendAlpha, 0.05f, 1.0f, "%.2f");
        ImGui::EndDisabled();
        ImGui::SliderInt("Refill passes", &p.maxRefillPasses, 0, 12);
    }

    if (ImGui::CollapsingHeader("Random"))
    {
        int seedI = (int)p.seed;
        if (ImGui::InputInt("Seed", &seedI)) p.seed = (uint32_t)seedI;
        ImGui::SameLine();
        if (ImGui::SmallButton("Randomize")) p.seed = (uint32_t)GetTickCount();
    }
}

// ---------- Main ----------
int main(int, char**)
{
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
                       GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                       L"ImageEditor", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"ImageEditor",
                                WS_OVERLAPPEDWINDOW, 100, 100, 1500, 900,
                                nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    Image       inputImg, outputImg;
    GpuTexture  inputTex, outputTex;
    std::wstring loadedPathW;
    std::string  loadedPathDisplay;
    std::string  statusMsg = "Ready.";
    BrushStrokeParams params;

    // Async stylize job (off the UI thread). The worker reads input/params from the
    // job snapshot (so the user can keep tweaking sliders while it runs), writes the
    // result, then sets `done`. Main thread polls `done` and consumes the result.
    struct StylizeJob {
        std::thread       worker;
        Image             input;
        BrushStrokeParams params;
        Image             result;
        StylizeContext    ctx;
        std::atomic<bool> done{ false };
        bool              running = false;
    } job;

    ImVec4 clear_color(0.10f, 0.10f, 0.12f, 1.0f);

    bool done = false;
    while (!done)
    {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Consume finished stylize job (if any) before drawing UI.
        if (job.running && job.done.load(std::memory_order_acquire))
        {
            if (job.worker.joinable()) job.worker.join();
            const bool wasCancelled = job.ctx.cancel.load();
            if (!wasCancelled && job.result.valid())
            {
                outputImg = std::move(job.result);
                outputTex.Upload(g_pd3dDevice, outputImg);
                statusMsg = "Processed.";
            }
            else
            {
                statusMsg = wasCancelled ? "Cancelled." : "Process failed.";
            }
            job.result.clear();
            job.running = false;
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGuiWindowFlags wflags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
                                | ImGuiWindowFlags_NoMove    | ImGuiWindowFlags_NoCollapse
                                | ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::Begin("##root", nullptr, wflags);

        // Toolbar
        ImGui::BeginDisabled(job.running);
        if (ImGui::Button("Load"))
        {
            std::wstring path;
            if (OpenFileDialog(hwnd, path))
            {
                Image img;
                if (LoadImageFromPath(path.c_str(), img))
                {
                    inputImg = std::move(img);
                    inputTex.Upload(g_pd3dDevice, inputImg);
                    outputImg.clear();
                    outputTex.Release();
                    loadedPathW = path;
                    loadedPathDisplay = WideToUtf8(path);
                    statusMsg = "Loaded.";
                }
                else statusMsg = "Failed to load image.";
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!inputImg.valid() || job.running);
        if (ImGui::Button("Process"))
        {
            job.input  = inputImg;
            job.params = params;
            job.result.clear();
            job.ctx.progress.store(0.f);
            job.ctx.cancel.store(false);
            job.done.store(false);
            job.running = true;
            job.worker = std::thread([&job]() {
                job.result = Stylize(job.input, job.params, &job.ctx);
                job.done.store(true, std::memory_order_release);
            });
            statusMsg = "Processing...";
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!outputImg.valid() || job.running);
        if (ImGui::Button("Save"))
        {
            std::wstring path;
            if (SaveFileDialog(hwnd, path))
                statusMsg = SaveImageToPath(path.c_str(), outputImg) ? "Saved." : "Save failed.";
        }
        ImGui::EndDisabled();

        if (job.running)
        {
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) job.ctx.cancel.store(true);
            ImGui::SameLine();
            ImGui::ProgressBar(job.ctx.progress.load(), ImVec2(200.f, 0.f));
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("%s", statusMsg.c_str());
        if (!loadedPathDisplay.empty())
        {
            ImGui::SameLine();
            ImGui::TextDisabled(" (%s  %dx%d)", loadedPathDisplay.c_str(), inputImg.w, inputImg.h);
        }

        ImGui::Separator();

        const float availH = ImGui::GetContentRegionAvail().y;

        // Settings sidebar
        ImGui::BeginChild("##settings", ImVec2(280.f, availH), ImGuiChildFlags_Borders);
        DrawSettingsPanel(params);
        ImGui::EndChild();
        ImGui::SameLine();

        // Two-pane viewer
        ImGui::BeginChild("##panes", ImVec2(0, availH));
        const float halfW = ImGui::GetContentRegionAvail().x * 0.5f - 4.f;
        ImGui::BeginChild("##left", ImVec2(halfW, 0));
        DrawImagePane("Input", inputTex, "Click [Load] to pick an image.");
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("##right", ImVec2(0, 0));
        DrawImagePane("Output", outputTex, "Click [Process] after loading an image.");
        ImGui::EndChild();
        ImGui::EndChild();

        ImGui::End();

        // Render
        ImGui::Render();
        const float clear[4] = { clear_color.x, clear_color.y, clear_color.z, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0);
    }

    // Make sure the stylize worker is finished before we tear down resources.
    if (job.running)
    {
        job.ctx.cancel.store(true);
        if (job.worker.joinable()) job.worker.join();
        job.running = false;
    }

    inputTex.Release();
    outputTex.Release();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

// ---------- D3D plumbing ----------
static bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        createDeviceFlags, featureLevelArray, _countof(featureLevelArray),
        D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (hr == DXGI_ERROR_UNSUPPORTED)
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
            createDeviceFlags, featureLevelArray, _countof(featureLevelArray),
            D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (FAILED(hr)) return false;

    CreateRenderTarget();
    return true;
}

static void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain)        { g_pSwapChain->Release();        g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)        { g_pd3dDevice->Release();        g_pd3dDevice = nullptr; }
}

static void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    if (pBackBuffer)
    {
        g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
        pBackBuffer->Release();
    }
}

static void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED) return 0;
        g_ResizeWidth  = (UINT)LOWORD(lParam);
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
