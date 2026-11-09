#pragma once

namespace battery::ui {
// Optional in-process desktop frontend. Must run on the window-system thread.
// No UI type leaks into core/run/job. A remote service wraps those layers instead.
int run(int argc, char** argv);
} // namespace battery::ui
