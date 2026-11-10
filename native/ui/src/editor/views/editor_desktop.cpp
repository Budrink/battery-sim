#include "editor/views/editor_desktop.hpp"

namespace battery::ui {
EditorDesktop::EditorDesktop(DesktopHost& main, bool hidden)
    : main_(main), window_(hidden, &main), editor_(std::make_unique<ExperimentEditor>()) {
    main_.activate();
}
EditorDesktop::~EditorDesktop() {
    window_.activate();
    editor_.reset(); // release preview VAOs/textures in their creating context
    main_.activate();
}
void EditorDesktop::show() {
    editor_->show();
}
void EditorDesktop::draw(const std::string& capture) {
    try {
        if (visible_ && window_.should_close())
            editor_->open = false;
        if (editor_->open) {
            if (!visible_ || editor_->focus_requested) {
                window_.show();
                window_.focus();
                editor_->focus_requested = false;
            }
            visible_ = true;
            window_.begin_frame();
            editor_->draw();
            window_.present(capture);
            if (editor_->viewer_focus_requested) {
                editor_->viewer_focus_requested = false;
                main_.focus();
            }
        }
        if (!editor_->open && visible_) {
            window_.hide();
            visible_ = false;
        }
    } catch (...) {
        main_.activate();
        throw;
    }
    main_.activate();
}
} // namespace battery::ui
