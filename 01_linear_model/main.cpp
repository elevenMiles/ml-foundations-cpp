#include <chrono>
#include <vector>
#include <iostream>

struct Pair
{
	float x{};
	float y{};
};

class LinearModel
{
private:
	float w{ 0.78f };
	float b{ 1.0f };

public:
	LinearModel() = default;

	float get_w() const { return w; }
	float get_b() const { return b; }

	float forward(float x) const
	{
		return x * w + b;
	}

	float loss(const std::vector<Pair>& data_set) const
	{
		float total_loss{ 0.0f };
		for (const Pair& data : data_set)
		{
			float diff{ forward(data.x) - data.y };
			total_loss += diff * diff;
		}
		return total_loss;
	}

	void train_closedForm(const std::vector<Pair>& data_set)
	{
		// Compute dataset means.

		Pair mean{ 0.0f, 0.0f };

		for (const Pair& p : data_set)
		{
			mean.x += p.x;
			mean.y += p.y;
		}

		float data_count{ static_cast<float>(data_set.size()) };

		mean.x /= data_count;
		mean.y /= data_count;

		// Calculate variance and covariance accumulators.

		float variance{ 0.0f };
		float covariance{ 0.0f };

		for (const Pair& p : data_set)
		{
			float x_dev{ p.x - mean.x };
			float y_dev{ p.y - mean.y };

			covariance += x_dev * y_dev;
			variance += x_dev * x_dev;
		}

		// Calcualtion of optimal weight and bias terms.
		w = covariance / variance;
		b = mean.y - w * mean.x;
	}

	void train_gradientDescend(const std::vector<Pair>& data_set, float learning_rate, int epoch)
	{
		for (int i{ 0 }; i < epoch; ++i)
		{
			float w_grad{ 0.0f };
			float b_grad{ 0.0f };
			for (const Pair& p : data_set)
			{
				float diff{ forward(p.x) - p.y };
				w_grad += diff * p.x;
				b_grad += diff;
			}

			w -= 2.0f * w_grad * learning_rate;
			b -= 2.0f * b_grad * learning_rate;
		}
	}

	void reset()
	{
		*this = {};
	}
};

int main()
{
	// Preparation of dataset
	float slope{ 0.5f };
	float y_int{ 1.0f };

	std::vector<Pair> data_set;
	for (float x{ -2.0f }; x <= 2.0f; x += 0.5f)
	{
		data_set.push_back(Pair{ x, x * slope + y_int });
	}

	LinearModel linearModel;

	// Measurement of closed form training
	auto start_closed{ std::chrono::high_resolution_clock::now() };
	linearModel.train_closedForm(data_set);
	auto end_closed{ std::chrono::high_resolution_clock::now() };

	std::cout << "Closed form LOSS: " << linearModel.loss(data_set) << '\n';
	std::cout << "weight: " << linearModel.get_w() << ", bias: " << linearModel.get_b() << "\n";
	std::cout << '\n';

	linearModel.reset();
	// Measurement of gradient descent form training
	
	auto start_gradient{ std::chrono::high_resolution_clock::now() };
	linearModel.train_gradientDescend(data_set, 0.03f, 150);
	auto end_gradient{ std::chrono::high_resolution_clock::now() };

	std::cout << "Gradient descent LOSS: " << linearModel.loss(data_set) << '\n';
	std::cout << "weight: " << linearModel.get_w() << ", bias: " << linearModel.get_b() << "\n";
	std::cout << '\n';

	auto duration_closed_ns{ std::chrono::duration_cast<std::chrono::nanoseconds>(end_closed - start_closed) };
	auto duration_gradient_ns{ std::chrono::duration_cast<std::chrono::nanoseconds>(end_gradient - start_gradient) };

	std::cout << "Closed form training time: " << duration_closed_ns << '\n';
	std::cout << "Gradient descent training time: " << duration_gradient_ns << '\n';

	return 0;
}
