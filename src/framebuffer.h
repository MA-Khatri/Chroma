#pragma once

class Framebuffer
{
public:
	Framebuffer(unsigned int width, unsigned int height, bool depth_only = false);
	~Framebuffer();

	void Bind() const;
	void Unbind() const;

	void UpdateFramebufferSize(unsigned int width, unsigned int height);

	inline unsigned int GetTexture() { return m_RenderedTexture; }

private:
	unsigned int m_RenderedTexture;
};