#pragma once

#include "application.h"

class Layer
{
public:
	virtual ~Layer() = default;

	virtual void OnAttach() {}
	virtual void OnDetach() {}

	virtual void OnUpdate(float time_step) {}
	virtual void OnUIRender() {}
};