import gradio as gr


class TagInput(gr.HTML):
    def __init__(self, value=None, **kwargs):
        """
        Initializes a tag input component for dynamic tag management.

        This class provides an interactive user interface for creating, displaying,
        and removing tags. The HTML structure renders the tag input field along
        with any existing tags. Styles for the component are provided through the
        CSS template, and JavaScript is used to enable interactivity such as adding
        and removing tags.

        The class is taken from https://gradio.app/custom-components/html-gallery and was slightly modified.

        :param value: Initial list of tags to populate in the tag input component.
        :type value: list[str] or None
        :param kwargs: Additional properties to customize the underlying component.
        """
        html_template = """
        <div class="tag-input-container">
            <div class="tag-list">
                ${(value || []).map((tag, i) => `
                    <span class="tag-pill">
                        ${tag}
                        <button class="tag-remove" data-index="${i}">&times;</button>
                    </span>
                `).join('')}
                <input type="text" class="tag-text-input"
                       placeholder="Enter a Milestone and press Enter..." />
            </div>
        </div>
        """
        css_template = """
        .tag-list {
            display: flex; flex-wrap: wrap; gap: 6px;
            padding: 8px; border: 1px solid #e5e7eb;
            border-radius: 8px;
        }
        .tag-pill {
            padding: 4px 10px; background: #fff7ed;
            color: #ea580c; border-radius: 16px;
            font-size: 13px; border: 1px solid #fed7aa;
        }
        """
        js_on_load = """
element.addEventListener('keydown', (e) => {
    const input = element.querySelector('.tag-text-input');
    if (!input) return;

    if (e.target === input && e.key === 'Enter' && input.value.trim()) {
        e.preventDefault();
        const v = parseInt(input.value.trim(), 10);
if (!Number.isNaN(v)) {
    props.value = [...(props.value || []), v];
}
        input.value = '';
    }
});

element.addEventListener('click', (e) => {
    const btn = e.target.closest('.tag-remove');
    if (!btn) return;

    const idx = parseInt(btn.dataset.index);
    const tags = [...(props.value || [])];
    tags.splice(idx, 1);
    props.value = tags;
});
        """
        super().__init__(
            value=value or [],
            html_template=html_template,
            css_template=css_template,
            js_on_load=js_on_load, **kwargs
        )

    def api_info(self):
        return {"type": "array", "items": {"type": "integer"}}
