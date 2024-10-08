import * as cdk from '@aws-cdk/core';
import * as cloudwatch from '@aws-cdk/aws-cloudwatch';
import * as lambda from '@aws-cdk/aws-lambda';

export class MonitoringStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create a dashboard
    const dashboard = new cloudwatch.Dashboard(this, 'LoanSystemDashboard');

    // Add widgets to the dashboard
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Lambda Errors',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/Lambda',
            metricName: 'Errors',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
          }),
        ],
      }),
      new cloudwatch.GraphWidget({
        title: 'API Gateway Latency',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/ApiGateway',
            metricName: 'Latency',
            statistic: 'Average',
            period: cdk.Duration.minutes(5),
          }),
        ],
      }),
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Read/Write Capacity',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/DynamoDB',
            metricName: 'ConsumedReadCapacityUnits',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
          }),
          new cloudwatch.Metric({
            namespace: 'AWS/DynamoDB',
            metricName: 'ConsumedWriteCapacityUnits',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
          }),
        ],
      }),
      new cloudwatch.GraphWidget({
        title: 'SageMaker Endpoint Invocations',
        left: [
          new cloudwatch.Metric({
            namespace: 'AWS/SageMaker',
            metricName: 'Invocations',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
          }),
        ],
      })
    );

    // Create alarms
    new cloudwatch.Alarm(this, 'LambdaErrorAlarm', {
      metric: new cloudwatch.Metric({
        namespace: 'AWS/Lambda',
        metricName: 'Errors',
        statistic: 'Sum',
        period: cdk.Duration.minutes(5),
      }),
      threshold: 5,
      evaluationPeriods: 1,
      alarmDescription: 'Alarm if the SUM of Errors is greater than 5 for 5 minutes',
    });

    new cloudwatch.Alarm(this, 'APILatencyAlarm', {
      metric: new cloudwatch.Metric({
        namespace: 'AWS/ApiGateway',
        metricName: 'Latency',
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
      }),
      threshold: 1000, // 1 second
      evaluationPeriods: 3,
      alarmDescription: 'Alarm if the average API latency is greater than 1 second for 15 minutes',
    });
  }
}